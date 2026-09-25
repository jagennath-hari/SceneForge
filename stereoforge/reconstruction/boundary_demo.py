"""Bounded map-based registration from saved hierarchical tracks and one anchor."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

from stereoforge.geometry.config import RefinementConfig
from stereoforge.refinement.bundle_adjustment import SparseBundleAdjuster, require_connected, model_statistics
from stereoforge.refinement.sparse_model import SparseCamera, SparseModel
from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import Progress, progress_group
from stereoforge.utils.visualization import GeometryReportWriter
from .map_registration import MapRegistration, key


class BoundaryReconstructor(MapRegistration):
    def __init__(self, run: Path, node_id: int, extension: int, output: Path, device: int) -> None:
        self.run, self.output = run, output
        nodes = {}

        def visit(node: dict) -> None:
            nodes[node['id']] = node
            for child in node['children']:
                visit(child)

        visit(json.loads((run / 'hierarchy.json').read_text()))
        node = nodes[node_id]
        if len(node['children']) != 2:
            raise ValueError('Select an internal node with two completed child maps')
        left, right = node['children']
        models = []
        for child in (left, right):
            directory = run / 'nodes' / str(child['id']) / 'ba'
            if not (directory / 'complete.json').is_file():
                raise ValueError(f'Child {child["id"]} has no completed BA')
            models.append(SparseModel.read(directory / 'workspace/sparse', child['frames']))
        anchor, neighbor = models
        shared = sorted(set(left['frames']) & set(right['frames']))
        if not shared:
            raise ValueError('Children have no shared frames')
        # Remove overlap poses AND their observations before PnP. Otherwise a
        # camera could simply be recovered from landmarks that used that camera.
        fixed_ids = {f for f in anchor.cameras if f < min(shared)}
        points = []
        for point in anchor.points:
            observations = {f: uv for f, uv in point.observations.items() if f in fixed_ids}
            if len(observations) >= 3:
                points.append(replace(point, observations=observations))
        supported = {f for p in points for f in p.observations}
        if supported != fixed_ids or len(points) < 30:
            raise ValueError('Removing overlap leaves insufficient anchor support; no reconstruction started')
        self.model = SparseModel({f: anchor.cameras[f] for f in sorted(fixed_ids)}, tuple(points))
        require_connected(self.model)
        extension_ids = [f for f in right['frames'] if f > max(shared)][:extension]
        self.targets = shared + extension_ids
        self.calibration = {}
        for frame in self.targets:
            if frame not in neighbor.cameras:
                raise ValueError(f'No saved calibration for target {frame}')
            camera = neighbor.cameras[frame]
            self.calibration[frame] = SparseCamera(np.eye(4), camera.intrinsics.copy(), camera.size_hw)
        self.ids = sorted(fixed_ids | set(self.targets))
        self.tracks = []
        self.track_ids = []
        allowed = set(self.ids)
        with (run / 'tracks.jsonl').open() as stream:
            for line in stream:
                record = json.loads(line)
                track = {int(f): np.asarray(uv, dtype=float) for f, uv in record['observations'].items() if int(f) in allowed}
                if len(track) >= 3:
                    self.tracks.append(track)
                    self.track_ids.append(record['id'])
        self.lookup = {}
        for track_id, track in enumerate(self.tracks):
            for frame, uv in track.items():
                self.lookup.setdefault(key(frame, uv), set()).add(track_id)
        self.holdouts: dict[int, list[int]] = {}
        self.excluded: set[tuple[int, int]] = set()
        self.optimizer = SparseBundleAdjuster(RefinementConfig(device=device))
        self.package = self.optimizer.preflight()
        self.report = {'status': 'running', 'node': node_id, 'children': [left['id'], right['id']],
                       'anchor_frames': sorted(fixed_ids), 'shared_frames': shared, 'target_frames': self.targets,
                       'attempts': [], 'ba_rounds': [], 'units': 'reconstruction_units',
                       'policy': 'Saved tracks only; no neighbor poses/points used. First camera fixed during BA; other cameras may refine.',
                       'validation_scope': 'Target observations held out of this experiment PnP, triangulation and BA; anchor geometry was previously optimized using overlap images.'}
        self.save_report()

    def execute(self) -> dict:
        ba_round = 0
        with Progress('Registering boundary cameras', len(self.targets), 'frame') as progress:
            for pass_id in range(3):
                accepted = 0
                for frame in self.targets:
                    if frame in self.model.cameras:
                        continue
                    progress.status(f'frame {frame}; pass {pass_id+1}/3')
                    if self.register(frame, pass_id):
                        accepted += 1
                        progress.advance()
                        new_points = self.extend_tracks()
                        seed = self.model
                        directory = self.output / f'ba_{ba_round:03d}'
                        ba_round += 1
                        optimized = self.optimizer.optimize_sparse(seed, directory, self.ids, self.package)
                        lost = sorted(seed.cameras.keys() - optimized.cameras.keys())
                        self.model = optimized
                        validation = self.validate()
                        self.report['ba_rounds'].append({'directory': directory.name, 'added_points': new_points,
                                                         'lost_cameras': lost, 'validation': validation})
                        self.save_report()
                        if lost or not validation['passed']:
                            raise ValueError('Boundary BA lost supported cameras or failed held-out projections; inspect report.json')
                if all(f in self.model.cameras for f in self.targets) or not accepted:
                    break
        self.report['status'] = 'complete' if all(f in self.model.cameras for f in self.targets) else 'stalled'
        self.model.write(self.output / 'sparse', self.ids)
        xyz = np.array([p.xyz for p in self.model.points])
        rgb = np.array([p.rgb for p in self.model.points], dtype=np.uint8)
        GeometryReportWriter._write_ply(self.output / 'point_cloud.ply', xyz, rgb)
        frames = [{"frame_index": f, "source": f"Keyframe {f}", "processed_size_hw": c.size_hw,
                   "camera_to_world": c.pose.tolist(), "intrinsics": c.intrinsics.tolist(),
                   "valid_fraction": 0, "depth_p50": None,
                   "previews": {"rgb": os.path.relpath(self.run / 'processed' / f'{f:06d}.png', self.output)}}
                  for f, c in sorted(self.model.cameras.items())]
        metadata = {"format_version": 1, "units": "reconstruction_units", "meters_per_unit": None,
                    "reconstruction_name": "Boundary registration experiment", "sparse_only": True,
                    "dense_depth_refined": False, "frames": frames, "input_frames": len(self.ids),
                    "sparse_statistics": {"status": self.report['status'], **model_statistics(self.model)}}
        order = {f: i for i, f in enumerate(sorted(self.model.cameras))}
        appeared = np.array([min(order[f] for f in p.observations) for p in self.model.points])
        write_json(self.output / 'metadata.json', metadata)
        GeometryReportWriter._write_viewer(self.output, metadata, xyz, rgb, appeared)
        self.save_report()
        return self.report



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--node', type=int, default=34)
    parser.add_argument('--extension', type=int, default=16, help='Additional frames after the shared boundary (0–32)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if not 0 <= args.extension <= 32 or args.device < 0:
        parser.error('Use extension 0–32 and a nonnegative device index')
    output = None
    worker = None
    try:
        if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
            raise ValueError('Requested BA device is not visible')
        run = args.run.expanduser().resolve()
        output = run / datetime.now(timezone.utc).strftime('boundary_%Y%m%d_%H%M%S_%f')
        output.mkdir()
        worker = BoundaryReconstructor(run, args.node, args.extension, output, args.device)
        with progress_group('Boundary reconstruction'):
            report = worker.execute()
        logging.info('%s: %d/%d target cameras. Inspect %s', report['status'],
                     len(report['registered_targets']), len(report['target_frames']), output)
        return 0 if report['status'] == 'complete' else 1
    except (Exception, KeyboardInterrupt) as exc:
        if worker is not None:
            worker.report.update(status='failed', error=str(exc) or 'Interrupted')
            worker.save_report()
        logging.error('%s\nArtifacts retained in %s', str(exc) or 'Interrupted', output)
        if args.debug:
            logging.exception('Boundary reconstruction traceback')
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
