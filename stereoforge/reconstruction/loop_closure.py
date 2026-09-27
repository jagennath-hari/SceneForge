"""Retrieve revisits, verify measured correspondences, and expose them to cuNLS."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import subprocess
import shutil
import sys
from typing import TYPE_CHECKING

import numpy as np

from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import Progress
from .view_graph import Observation, VerifiedGraph

if TYPE_CHECKING:
    from .frontend import ReconstructionFrontend


@dataclass(frozen=True, slots=True)
class LoopPolicy:
    candidates_per_frame: int = 3
    exclusion_frames: int = 64
    exclusion_seconds: float = 10.0
    region_radius: int = 16
    minimum_inliers: int = 40
    minimum_inlier_ratio: float = 0.3
    minimum_grid_cells: int = 6  # occupied cells in each image's 4x4 grid


class LoopClosure:
    """No descriptor similarity is itself a geometric constraint.

    Run after temporal repair so long-range tracks cannot mask weak temporal
    boundaries. Preserve native feature IDs on the original feature-image grid.
    """

    def __init__(self, frontend: ReconstructionFrontend) -> None:
        self.frontend = frontend
        self.root = frontend.output
        self.folder = self.root / 'loop_closure'
        self.folder.mkdir(exist_ok=True)
        self.policy = LoopPolicy(exclusion_frames=max(64, frontend.options.window_size))
        self.evidence: dict[int, list[tuple[int, int]]] = {}
        self.summary: dict = {'enabled': True, 'policy': asdict(self.policy)}

    def prepare(self, paths: list[Path], timestamps: tuple) -> dict:
        shutil.copyfile(self.root / 'connection_repair/temporal_tracks.jsonl', self.root / 'global_tracks.jsonl')
        visualization = self.frontend.visualization
        if visualization is not None:
            visualization.event('Retrieving possible revisits with SelaVPR++; candidates require geometric verification')
        request = self.folder / 'descriptor_request.json'
        descriptors = self.folder / 'descriptors.npy'
        write_json(request, {'paths': [str(p) for p in paths], 'device': self.frontend.device,
                             'output': str(descriptors)})
        # Separate interpreter isolates upstream's generic `model` imports and
        # releases descriptor inference memory before TensorRT matching/VGGT.
        subprocess.run([sys.executable, '-m', 'stereoforge.reconstruction.retrieval', str(request)], check=True)
        pairs, primary = self._retrieve(np.load(descriptors, allow_pickle=False), timestamps)
        write_json(self.folder / 'candidates.json', {'primary': sorted(primary), 'verification_pairs': pairs})
        self.summary.update(retrieved_pairs=len(primary), verification_pairs=len(pairs))
        if not pairs:
            self.summary.update(verified_pairs=0, tracks_reaching_track_builder=0)
            self._report()
            return self.summary
        try:
            self.frontend.output = self.folder
            files = self.frontend._match(len(paths), description='Verifying loop candidates', explicit_pairs=pairs)
        finally:
            self.frontend.output = self.root
        accepted = self._verify(files)
        accepted_file = self.folder / 'verified.jsonl'
        with accepted_file.open('w') as stream:
            for record in accepted:
                stream.write(json.dumps(record) + '\n')
        self.summary['verified_pairs'] = len(accepted)
        if not accepted:
            self.summary['tracks_reaching_track_builder'] = 0
            self._report()
            return self.summary
        repaired = json.loads((self.root / 'connection_repair/complete.json').read_text())
        last_round = len(repaired['rounds']) - 1
        temporal = sorted((self.root / f'connection_repair/round_{last_round:02d}/matching').glob('*.jsonl'))
        if not temporal:
            raise FileNotFoundError('Final temporal matching records are missing')
        graph = VerifiedGraph(len(paths), retain_feature_tracks=True)
        graph.read([*temporal, accepted_file],
                   on_progress=visualization.event if visualization else None,
                   on_tracks=visualization.tracks if visualization else None,
                   on_orbit=visualization.orbit if visualization else None,
                   description='Joining temporal and loop tracks')
        evidence: dict[int, set[tuple[int, int]]] = {}
        for record in accepted:
            a, b = record['source'], record['target']
            for match in record['matches']:
                if not match['inlier']:
                    continue
                left = graph.observation_tracks.get(Observation(a, int(match['source_feature'])))
                right = graph.observation_tracks.get(Observation(b, int(match['target_feature'])))
                if left is not None and left == right:
                    evidence.setdefault(left, set()).add((a, b))
        self.evidence = {track: sorted(edges) for track, edges in evidence.items()}
        temporary = self.root / 'global_tracks.partial'
        with temporary.open('w') as stream:
            for track in graph.tracks:
                stream.write(json.dumps({str(f): uv.tolist() for f, uv in track.items()}) + '\n')
        temporary.replace(self.root / 'global_tracks.jsonl')
        write_json(self.root / 'graph.json', graph.summary)
        write_json(self.folder / 'track_evidence.json', self.evidence)
        self.summary.update(tracks_reaching_track_builder=len(self.evidence),
                            combined_tracks=len(graph.tracks),
                            conflicting_edges_rejected=graph.summary['conflicting_edges_rejected'])
        self._report()
        return self.summary

    def _retrieve(self, descriptors: np.ndarray, timestamps: tuple) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
        count = len(descriptors)
        if descriptors.shape != (count, 2048) or len(timestamps) != count or not np.isfinite(descriptors).all():
            raise ValueError('Invalid descriptor matrix/timestamp association')
        times = np.asarray(timestamps, dtype=np.float64)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Loop retrieval requires strictly increasing timestamps')
        indices = np.arange(count)
        primary: set[tuple[int, int]] = set()
        with Progress('Retrieving distant image pairs', count, 'frame') as progress:
            # BLAS batches bound memory; float descriptors are already normalized.
            for start in range(0, count, 128):
                similarities = descriptors[start:start+128] @ descriptors.T
                for offset, scores in enumerate(similarities):
                    source = start + offset
                    valid = ((abs(indices-source) >= self.policy.exclusion_frames) &
                             (abs(times-times[source]) >= self.policy.exclusion_seconds))
                    selected = []
                    for target in np.argsort(-scores, kind='stable'):
                        target = int(target)
                        if not valid[target] or any(abs(target-other) <= self.policy.region_radius for other in selected):
                            continue
                        primary.add(tuple(sorted((source, target))))
                        selected.append(target)
                        if len(selected) == self.policy.candidates_per_frame:
                            break
                    progress.advance()
        pairs = set(primary)
        # Explicitly test neighboring frames in both traversal directions.
        for a, b in primary:
            for da, db in ((1, 1), (1, -1)):
                x, y = a+da, b+db
                if (0 <= x < y < count and y-x >= self.policy.exclusion_frames
                        and times[y]-times[x] >= self.policy.exclusion_seconds):
                    pairs.add((x, y))
        return sorted(pairs), primary

    def _verify(self, files: list[Path]) -> list[dict]:
        width, height = self.frontend.size_wh
        good = {}
        rejected = {'inliers': 0, 'coverage': 0, 'neighbor_agreement': 0}
        for file in files:
            with file.open() as stream:
                for line in stream:
                    record = json.loads(line)
                    matches = [m for m in record['matches'] if m['inlier']]
                    if (len(matches) < self.policy.minimum_inliers or
                            len(matches) < self.policy.minimum_inlier_ratio * len(record['matches'])):
                        rejected['inliers'] += 1
                        continue
                    distributed = True
                    for key in ('a', 'b'):
                        uv = np.asarray([m[key] for m in matches], dtype=np.float64)
                        if (uv.shape != (len(matches), 2) or not np.isfinite(uv).all() or
                                np.any(uv < 0) or np.any(uv >= [width, height])):
                            distributed = False
                            break
                        cells = np.floor(uv / [width, height] * 4).astype(int)
                        if len(np.unique(cells, axis=0)) < self.policy.minimum_grid_cells:
                            distributed = False
                    if not distributed:
                        rejected['coverage'] += 1
                        continue
                    good[record['source'], record['target']] = record
        accepted = []
        for (a, b), record in good.items():
            # Require a second geometrically verified nearby pair with BOTH
            # endpoints changed. A single repeated texture is insufficient.
            supported = any((a+da, b+db) in good for da in (-1, 1) for db in (-1, 1))
            if supported:
                accepted.append(record)
            else:
                rejected['neighbor_agreement'] += 1
        self.summary['rejected_pairs'] = rejected
        return accepted

    def measure(self, model, label: str) -> None:
        """Count actual native landmarks, not retrieved or merely matched pairs."""
        landmarks = model.landmarks
        tracks, pairs = 0, set()
        for identifier, edges in self.evidence.items():
            point = landmarks.get(identifier)
            if point is None:
                continue
            observations = point.observations
            surviving = [(a, b) for a, b in edges if a in observations and b in observations]
            if surviving:
                tracks += 1
                pairs.update(surviving)
        self.summary[label] = {'loop_landmarks': tracks, 'supported_loop_pairs': len(pairs)}
        self._report()

    def _report(self) -> None:
        write_json(self.folder / 'report.json', self.summary)
        logging.info('Loop closure: %d candidate pairs, %d verified pairs, %d consistent loop tracks',
                     self.summary.get('retrieved_pairs', 0), self.summary.get('verified_pairs', 0),
                     self.summary.get('tracks_reaching_track_builder', 0))
