"""Overlapping VGGT windows accumulated into one cuNLS-refined sparse map."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path

import numpy as np

from stereoforge.refinement.sparse_model import SparseModel
from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import Progress, progress_group
from stereoforge.utils.visualization import GeometryReportWriter
from .native_map import NativeMap
from .frontend import ReconstructionFrontend
from .inference import infer_clusters
from .view_graph import Cluster, VerifiedGraph

POLICY = 'native_windowed_cunls_v2'


@dataclass(frozen=True, slots=True)
class WindowOptions:
    window_size: int = 32
    overlap: int = 8
    neighbors: int = 4
    device: str = 'cuda'
    lm_iterations: int = 300

    def __post_init__(self) -> None:
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
                            error=error, dense_depth_refined=False,
                            reconstruction_name='StereoForge · VGGT windows + cuNLS',
                            units='reconstruction_units', attempt=str(self.attempt.relative_to(self.output)))
        self.summary['missing_frames'] = sorted(set(range(self.summary['input_frames'])) - (set(model.cameras) if model else set()))
        write_json(self.output / 'status.json', self.summary)
        if model is not None:
            # One viewer/trajectory for the current common map. A partial map
            # is explicitly labelled, never substituted for a completed scene.
            self._viewer(model, self.summary, self.timestamps)
            write_json(self.output / 'trajectory.json', {
                'status': status, 'convention': 'camera_to_world; x right, y down, z forward',
                'units': 'reconstruction_units', 'meters_per_unit': None,
                'frames': [{'keyframe_index': f,
                            'timestamp_seconds': self.timestamps[f] if self.timestamps else None,
                            'camera_to_world': c.pose.tolist(), 'intrinsics': c.intrinsics.tolist(),
                            'image_size_hw': c.size_hw} for f, c in sorted(model.cameras.items())]})
            # The viewer may subsample large clouds; export every optimized point.
            GeometryReportWriter._write_ply(self.output / 'point_cloud.ply',
                np.asarray([p.xyz for p in model.points]), np.asarray([p.rgb for p in model.points], dtype=np.uint8))
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
                graph.read(files)
                write_json(self.output / 'graph.json', graph.summary)
                temporary = tracks.with_suffix('.partial')
                with temporary.open('w') as stream:
                    for track in graph.tracks:
                        stream.write(json.dumps({str(f): uv.tolist() for f, uv in track.items()})+'\n')
                temporary.replace(tracks)
            write_json(self.output / 'windows.json', {'windows': [window.document() for window in windows]})
            self.summary['stage'] = 'vggt'
            infer_clusters(checkpoint, paths, windows, self.output / 'vggt', self.devices)
            with Progress('Loading native feature tracks'):
                self.native.load_tracks(tracks)
            with progress_group('Building common map') as progress:
                for index, window in enumerate(windows):
                    self.summary['stage'] = f'window_{index}'
                    def update(stage: str) -> None:
                        progress.status(f'window {index+1}/{len(windows)} | {stage}')
                    self.native.add_window(self.output / 'vggt' / f'{window.identifier}.pt', self.images, update)
                    self.summary['accepted_windows'] = self.native.accepted_windows
            self.current = self.native.export()
            if self.current is None or set(self.current.cameras) != set(range(len(paths))):
                raise ValueError('Common map does not contain every selected keyframe')
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
