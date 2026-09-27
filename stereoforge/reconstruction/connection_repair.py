"""Bounded insertion of decoded video frames at weak measured-track boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from stereoforge.utils.artifacts import write_json
from stereoforge.video.sampling import VideoFrameSampler
from .view_graph import VerifiedGraph

if TYPE_CHECKING:
    from .frontend import ReconstructionFrontend


@dataclass(frozen=True, slots=True)
class RepairPolicy:
    # This triggers extra measurements; it is not a geometry acceptance gate.
    minimum_crossing_tracks: int = 60
    maximum_rounds: int = 2
    maximum_added_fraction: float = 0.25
    maximum_added_frames: int = 512


class ConnectionRepair:
    """Keep original frames and matches, adding evidence before assigning windows.

    Each round owns its numbered images and matches, so inserting a frame cannot
    reuse a file under a stale frame ID. Existing pair records are explicitly
    remapped by immutable input path. Only new image pairs require inference.
    """

    def __init__(self, frontend: ReconstructionFrontend, policy: RepairPolicy = RepairPolicy()) -> None:
        self.frontend = frontend
        self.policy = policy
        self.root = frontend.output
        self.directory = self.root / 'connection_repair'
        self.directory.mkdir(exist_ok=True)

    def prepare(self, paths: list[Path], timestamps: tuple | None) -> tuple[list[Path], tuple]:
        complete = self.directory / 'complete.json'
        if complete.is_file():
            saved = json.loads(complete.read_text())
            paths = [Path(frame['path']) for frame in saved['frames']]
            self.frontend.images = {i: Path(path) for i, path in enumerate(saved['images'])}
            with Image.open(self.frontend.images[0]) as image:
                self.frontend.size_wh = image.size
            if any(not path.is_file() for path in [*paths, *self.frontend.images.values()]):
                raise FileNotFoundError('Repaired frame inputs are missing; start a fresh --video run')
            if not (self.root / 'global_tracks.jsonl').is_file():
                raise FileNotFoundError('Repaired feature tracks are missing; start a fresh --video run')
            if self.frontend.visualization is not None:
                self.frontend.visualization.keyframes(list(self.frontend.images.values()))
                self.frontend.visualization.event('Reusing repaired frame selection and measured tracks')
            return paths, tuple(frame['timestamp_seconds'] for frame in saved['frames'])

        manifest = json.loads((self.root / 'input_frames/manifest.json').read_text())
        if len(paths) != len(manifest['frames']):
            raise ValueError('Input paths do not match the selected-frame manifest')
        frames = [{**record, 'path': str(path)} for record, path in zip(manifest['frames'], paths, strict=True)]
        if timestamps is not None and tuple(frame['timestamp_seconds'] for frame in frames) != timestamps:
            raise ValueError('Input timestamps do not match the selected-frame manifest')
        cache_value = manifest.get('decoded_cache')
        if not cache_value:
            raise ValueError('Selected frames lack decoded-cache provenance; start a fresh --video run')
        cache = Path(cache_value)
        decoded = json.loads((cache / 'manifest.json').read_text())['frames']
        # Verify provenance before inserting any images. Candidate order is the
        # native decoder's timestamp order, never inferred from a filename.
        previous_time = -math.inf
        for index, record in enumerate(decoded):
            timestamp = record['timestamp_seconds']
            if not math.isfinite(timestamp) or timestamp <= previous_time:
                raise ValueError('Decoded candidates have invalid timestamps')
            previous_time = timestamp
            record['candidate_index'] = index
            record.setdefault('source_frame_index', index)
        for frame in frames:
            candidate = decoded[frame['candidate_index']]
            if any(frame[key] != candidate[key] for key in ('timestamp_seconds', 'source_frame_index', 'bytes')):
                raise ValueError('Selected frame does not match the decoded cache')

        initial_count = len(frames)
        budget = min(self.policy.maximum_added_frames,
                     math.ceil(initial_count * self.policy.maximum_added_fraction))
        previous_images: dict[Path, Path] = {}
        previous_files: list[Path] = []
        previous_paths: list[Path] = []
        history = []
        try:
            for round_index in range(self.policy.maximum_rounds + 1):
                folder = self.directory / f'round_{round_index:02d}'
                folder.mkdir(exist_ok=True)
                paths = [Path(frame['path']) for frame in frames]
                identity = {'paths': [str(path) for path in paths],
                            'candidate_indices': [frame['candidate_index'] for frame in frames]}
                identity_path = folder / 'inputs.json'
                if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
                    raise ValueError('Repair round input identity changed; start a fresh --video run')
                write_json(identity_path, identity)
                self.frontend.output = folder
                prefix = f"Repair {round_index}/{self.policy.maximum_rounds}"
                self.frontend.images = self.frontend._prepare_images(paths, previous_images,
                    description=f"{prefix} · preparing images" if round_index else "Preparing feature images")
                reused = self._remap_matches(previous_files, previous_paths, paths)
                files = self.frontend._match(len(paths), reused,
                    description=f"{prefix} · matching pairs" if round_index else "Verifying temporal image pairs")
                del reused
                graph = VerifiedGraph(len(paths))
                visualization = self.frontend.visualization
                graph.read(files, on_progress=visualization.event if visualization else None,
                           on_tracks=visualization.tracks if visualization else None,
                           on_orbit=visualization.orbit if visualization else None,
                           description=f"{prefix} · rebuilding tracks" if round_index else "Building measured feature tracks")
                counts = self._crossing_counts(graph)
                weak = [i for i in range(1, len(frames)) if counts[i] < self.policy.minimum_crossing_tracks]
                remaining = budget - (len(frames) - initial_count)
                additions = (self._candidates(frames, weak, counts, remaining)
                             if round_index < self.policy.maximum_rounds else [])
                report = {'round': round_index, 'frames': len(frames), 'tracks': len(graph.tracks),
                          'weak_boundaries': [{'left_source_frame': frames[i-1]['source_frame_index'],
                                               'right_source_frame': frames[i]['source_frame_index'],
                                               'crossing_tracks': int(counts[i])} for i in weak],
                          'inserted_candidate_indices': additions}
                history.append(report)
                write_json(folder / 'support.json', report)
                logging.info('Connection repair round %d: %d weak boundaries; %d intermediate frames to add',
                             round_index, len(weak), len(additions))
                if not additions:
                    if weak:
                        logging.warning('Connection repair finished with %d weak boundaries; geometry checks remain unchanged', len(weak))
                    self._publish(frames, graph, history, initial_count, budget)
                    logging.info('VGGT input: %d selected frames + %d repair frames = %d frames',
                                 initial_count, len(frames)-initial_count, len(frames))
                    return paths, tuple(frame['timestamp_seconds'] for frame in frames)
                previous_images = dict(zip(paths, self.frontend.images.values(), strict=True))
                previous_paths, previous_files = paths, files
                del graph
                candidate_folder = self.directory / 'candidates'
                candidate_folder.mkdir(exist_ok=True)
                for index in additions:
                    record = decoded[index]
                    source = cache / record['file']
                    if not source.is_file() or source.stat().st_size != record['bytes']:
                        raise ValueError(f'Decoded repair candidate is missing or changed: {source}')
                    retained = candidate_folder / f'{index:08d}.png'
                    if not retained.exists():
                        VideoFrameSampler._link_or_copy(source, retained)
                    if retained.stat().st_size != record['bytes']:
                        raise ValueError(f'Retained repair candidate changed: {retained}')
                    frames.append({**record, 'path': str(retained), 'selection_reason': 'weak_connection_repair'})
                frames.sort(key=lambda frame: frame['candidate_index'])
        finally:
            self.frontend.output = self.root
        raise RuntimeError('Connection repair did not publish a final selection')

    @staticmethod
    def _crossing_counts(graph: VerifiedGraph) -> np.ndarray:
        # A measured track contributes once to every temporal cut it crosses.
        delta = np.zeros(graph.count + 1, dtype=np.int64)
        for track in graph.tracks:
            delta[min(track) + 1] += 1
            delta[max(track) + 1] -= 1
        return np.cumsum(delta)[:graph.count]

    @staticmethod
    def _candidates(frames: list[dict], weak: list[int], counts: np.ndarray, budget: int) -> list[int]:
        selected: set[int] = set()
        # Start at the weakest cut, then fill its immediate neighboring gaps to
        # provide three-view tracks, not just one stronger two-view match.
        for boundary in sorted(weak, key=lambda index: (counts[index], index)):
            for right in (boundary, boundary-1, boundary+1):
                if len(selected) >= budget:
                    return sorted(selected)
                if not 0 < right < len(frames):
                    continue
                first, last = frames[right-1]['candidate_index'], frames[right]['candidate_index']
                if last - first > 1:
                    selected.add((first + last) // 2)
        return sorted(selected)

    @staticmethod
    def _remap_matches(files: list[Path], previous: list[Path], current: list[Path]) -> dict[tuple[int, int], dict]:
        indices = {path: index for index, path in enumerate(current)}
        result = {}
        for file in files:
            with file.open() as stream:
                for line in stream:
                    record = json.loads(line)
                    a, b = indices[previous[record['source']]], indices[previous[record['target']]]
                    record['source'], record['target'] = a, b
                    result[a, b] = record
        return result

    def _publish(self, frames: list[dict], graph: VerifiedGraph, history: list[dict], initial_count: int, budget: int) -> None:
        temporary = self.root / 'global_tracks.partial'
        with temporary.open('w') as stream:
            for track in graph.tracks:
                stream.write(json.dumps({str(frame): uv.tolist() for frame, uv in track.items()}) + '\n')
        temporary.replace(self.root / 'global_tracks.jsonl')
        shutil.copyfile(self.root / 'global_tracks.jsonl', self.directory / 'temporal_tracks.jsonl')
        write_json(self.root / 'graph.json', graph.summary)
        original_selection = json.loads((self.root / 'selection.json').read_text())
        write_json(self.root / 'selection.json', {
            'frames': len(frames), 'initial_keyframes': initial_count,
            'decoded_candidates': original_selection['decoded_candidates'],
            'source_frame_indices': [frame['source_frame_index'] for frame in frames],
            'timestamps_seconds': [frame['timestamp_seconds'] for frame in frames],
            'repair_added_frames': len(frames)-initial_count})
        complete = self.directory / 'complete.partial'
        write_json(complete, {
            'frames': frames, 'images': [str(path) for path in self.frontend.images.values()],
            'initial_keyframes': initial_count, 'insertion_budget': budget,
            'minimum_crossing_tracks': self.policy.minimum_crossing_tracks,
            'rounds': history})
        complete.replace(self.directory / 'complete.json')
