"""Overlapping VGGT windows accumulated into one cuNLS-refined sparse map."""

from dataclasses import dataclass
import json
import logging
import shutil
from pathlib import Path

import numpy as np

from stereoforge.refinement.sparse_model import SparseModel
from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import Progress, progress_group
from stereoforge.utils.point_cloud import write_ply
from .native_map import NativeMap, WindowRejected
from .dense import DenseRefiner, DenseOptions
from .frontend import ReconstructionFrontend
from .inference import infer_clusters
from .view_graph import Cluster, VerifiedGraph

POLICY = 'native_graph_dense_cunls_v9'


@dataclass(frozen=True, slots=True)
class WindowOptions:
    window_size: int = 32
    overlap: int = 16
    neighbors: int = 4
    device: str = 'cuda'
    lm_iterations: int = 500
    dense_voxel_fraction: float = 0.01

    def __post_init__(self) -> None:
        if not 0.001 <= self.dense_voxel_fraction <= 0.1:
            raise ValueError('Dense voxel fraction must be between 0.001 and 0.1')
        if self.window_size < 8 or not 6 <= self.overlap < self.window_size:
            raise ValueError('Require window-size >= 8 and 6 <= overlap < window-size')
        if not 1 <= self.neighbors <= 12 or not 1 <= self.lm_iterations <= 500:
            raise ValueError('Require 1–12 matching neighbors and 1–500 joint BA iterations')
        if self.device != 'cuda' and not (self.device.startswith('cuda:') and self.device[5:].isdigit()):
            raise ValueError('Use --device cuda or cuda:N')

    def windows(self, count: int) -> list[Cluster]:
        if count < 6:
            raise ValueError('At least six selected keyframes are required')
        result = []
        start = 0
        while True:
            stop = min(count, start+self.window_size)
            result.append(Cluster(len(result), list(range(start, stop))))
            if stop == count:
                return result
            start += self.window_size-self.overlap


class WindowReconstructor(ReconstructionFrontend):
    def __init__(self, options: WindowOptions, output: Path, keyframe_config: Path,
                 attempt: Path, diagnostics: bool = False) -> None:
        super().__init__(options, output, keyframe_config)
        self.native = NativeMap(self.device, options.lm_iterations, diagnostics)
        self.attempt, self.diagnostics = attempt, diagnostics
        self.current: SparseModel | None = None
        self.timestamps: tuple | None = None
        self.summary: dict = {}

    def publish(self, status: str, error: str | None = None) -> dict:
        model = self.current
        self.summary.update(native_stage_seconds=self.native.stage_seconds, status=status, registered_frames=len(model.cameras) if model else 0,
                            error=error, dense_depth_refined=bool(self.summary.get("dense_refinement")),
                            shared_calibration_complete=self.native.builder.shared_calibration_complete,
                            native_stage=self.native.last_stage,
                            alignment_warnings=self.native.alignment_warnings,
                            alignment_confidence_warning=bool(self.native.alignment_warnings),
                            reconstruction_name='StereoForge · VGGT windows + cuNLS',
                            units='reconstruction_units', attempt=str(self.attempt.relative_to(self.output)))
        if model is not None and self.native.builder.shared_calibration_complete:
            self.summary['shared_intrinsics'] = next(iter(model.cameras.values())).intrinsics.tolist()
        self.summary['missing_frames'] = sorted(set(range(self.summary['input_frames'])) - (set(model.cameras) if model else set()))
        write_json(self.output / 'status.json', self.summary)
        if model is not None:
            write_json(self.output / 'trajectory.json', {
                'status': status, 'convention': 'camera_to_world; x right, y down, z forward',
                'units': 'reconstruction_units', 'meters_per_unit': None,
                'frames': [{'keyframe_index': f,
                            'timestamp_seconds': self.timestamps[f] if self.timestamps else None,
                            'camera_to_world': c.pose.tolist(), 'intrinsics': c.intrinsics.tolist(),
                            'image_size_hw': c.size_hw} for f, c in sorted(model.cameras.items())]})
            # The viewer may subsample large clouds; export every optimized point.
            write_ply(self.output / 'point_cloud.ply',
                np.asarray([p.xyz for p in model.points]), np.asarray([p.rgb for p in model.points], dtype=np.uint8))
            if self.summary.get('dense_refinement'):
                # Preserve sparse output alongside the validated dense cloud. Raw VGGT windows are never overwritten.
                (self.output / 'point_cloud.ply').replace(self.output / 'sparse_point_cloud.ply')
                shutil.copyfile(self.output / 'dense_point_cloud.ply', self.output / 'point_cloud.ply')
        return self.summary

    def run(self, paths: list[Path], checkpoint: Path, timestamps: tuple | None) -> dict:
        self.timestamps = timestamps
        windows = self.options.windows(len(paths))
        self.summary = {'input_frames': len(paths), 'windows': len(windows),
                        'accepted_windows': 0, 'stage': 'features'}
        try:
            self.images = self._prepare_images(paths)
            tracks = self.output / 'global_tracks.jsonl'
            if not tracks.exists():
                graph = VerifiedGraph(len(paths))
                files = self._match(len(paths))
                if self.visualization is not None:
                    self.visualization.event('Building measured feature tracks from verified image matches')
                graph.read(files, on_progress=self.visualization.event if self.visualization is not None else None,
                           on_tracks=self.visualization.tracks if self.visualization is not None else None,
                           on_orbit=self.visualization.orbit if self.visualization is not None else None)
                if self.visualization is not None:
                    self.visualization.event(f"Built {len(graph.tracks)} measured tracks; keyframe positions remain display-only until VGGT")
                write_json(self.output / 'graph.json', graph.summary)
                temporary = tracks.with_suffix('.partial')
                with temporary.open('w') as stream:
                    for track in graph.tracks:
                        stream.write(json.dumps({str(f): uv.tolist() for f, uv in track.items()})+'\n')
                temporary.replace(tracks)
            elif self.visualization is not None:
                self.visualization.event('Reusing saved measured feature tracks; matching and track building skipped')
            write_json(self.output / 'windows.json', {'windows': [window.document() for window in windows]})
            self.summary['stage'] = 'vggt'
            on_window = None
            if self.visualization is not None:
                self.visualization.event('VGGT inference: waiting for the first window; local maps remain separate until Sim(3)')
                def on_window(identifier: int) -> None:
                    self.visualization.window(self.output / 'vggt' / f'{identifier}.pt', list(self.images.values()), identifier)
            infer_clusters(checkpoint, paths, windows, self.output / 'vggt', self.devices, on_complete=on_window)
            with Progress('Loading native feature tracks'):
                self.native.load_tracks(tracks)
            pending = {window.identifier: window for window in windows}
            # Frame ownership supplies accepted raw depths in the appropriate
            # gauge, independent of processing order. Tensors remain disk-cached.
            owners: dict[int, Path] = {}
            attempted_at: dict[int, int] = {}
            reasons: dict[int, str] = {}
            waiting_for: dict[int, frozenset[int]] = {}
            self.summary['merge_order'] = []
            self.summary['unresolved_windows'] = []
            self.summary['global_ba_complete'] = False
            with progress_group('Building common map', len(windows), 'window') as progress:
                while pending:
                    revision = self.native.accepted_windows
                    eligible = [window for key, window in sorted(pending.items())
                                if attempted_at.get(key) != revision
                                and (key not in waiting_for or not waiting_for[key].isdisjoint(owners))]
                    ranked = self.native.rank_windows([window.frames for window in eligible])
                    if not ranked:
                        break
                    window = eligible[ranked[0]]
                    index = window.identifier
                    self.summary['stage'] = f'window_{index}'
                    def update(stage: str) -> None:
                        progress.status(f'window {index+1}/{len(windows)} | {stage}')
                    references = {frame: owners[frame] for frame in window.frames if frame in owners}
                    source = self.output / 'vggt' / f'{index}.pt'
                    try:
                        self.native.add_window(source, self.images, references, update)
                    except WindowRejected as error:
                        attempted_at[index] = revision
                        reasons[index] = str(error)
                        if error.unanchored_frames:
                            waiting_for[index] = error.unanchored_frames
                        else:
                            waiting_for.pop(index, None)
                        logging.warning('Deferred window %d at map revision %d: %s', index, revision, error)
                        continue
                    for frame in window.frames:
                        owners.setdefault(frame, source)
                    del pending[index]
                    reasons.pop(index, None)
                    waiting_for.pop(index, None)
                    self.summary['merge_order'].append(index)
                    self.summary['accepted_windows'] = self.native.accepted_windows
                    progress.advance()
                self.summary['unresolved_windows'] = [
                    {'window': key, 'frames': window.frames,
                     'reason': reasons.get(key, 'No shared camera with the accepted map'),
                     'last_attempt_revision': attempted_at.get(key),
                     'waiting_for_shared_cameras': sorted(waiting_for.get(key, ()))}
                    for key, window in sorted(pending.items())]
            if not self.native.accepted_windows:
                raise RuntimeError('No window could initialize a connected map; inspect unresolved_windows in status.json')
            self.summary['stage'] = 'global_ba'
            with Progress('Final global bundle adjustment') as progress:
                self.native.finalize(lambda stage: progress.status(stage))
            self.summary['global_ba_complete'] = True
            self.current = self.native.export()
            if self.current is None or set(self.current.cameras) != set(range(len(paths))):
                self.summary['stage'] = 'unresolved_windows'
                return self.publish('partial', 'No further connected windows could be accepted; inspect unresolved_windows')
            self.summary['stage'] = 'dense_refinement'
            self.summary['dense_refinement'] = DenseRefiner(self.output,self.device, DenseOptions(voxel_depth_fraction=self.options.dense_voxel_fraction)).run(self.current,owners, visualization=self.visualization)
            self.summary['stage'] = 'finished'
            return self.publish('complete')
        except (Exception, KeyboardInterrupt) as error:
            self.current = self.native.export()
            self.summary['accepted_windows'] = self.native.accepted_windows
            try:
                self.publish('partial' if self.current else 'failed', str(error) or 'Interrupted')
            except Exception:
                logging.exception('Could not publish the partial-map report; preserving original failure')
            raise
