"""Bounded map-based registration from saved hierarchical tracks and one anchor."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from stereoforge.geometry.config import RefinementConfig
from stereoforge.refinement.bundle_adjustment import SparseBundleAdjuster, require_connected, model_statistics
from stereoforge.refinement.sparse_model import SparseCamera, SparseModel, SparsePoint, reprojection_error
from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import Progress, progress_group
from stereoforge.utils.visualization import GeometryReportWriter
from .triangulation import TrackTriangulator


def key(frame: int, uv: np.ndarray) -> tuple:
    return frame, *map(float, np.round(uv, 4))


class BoundaryReconstructor:
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

    def associations(self) -> dict[int, int]:
        proposed = {}
        for point_id, point in enumerate(self.model.points):
            votes = {}
            for frame, uv in point.observations.items():
                for track_id in self.lookup.get(key(frame, uv), ()):
                    votes[track_id] = votes.get(track_id, 0) + 1
            if len(votes) == 1:
                track_id, count = next(iter(votes.items()))
                if count >= 2:
                    proposed.setdefault(track_id, []).append(point_id)
        return {t: ids[0] for t, ids in proposed.items() if len(ids) == 1}

    def register(self, frame: int, pass_id: int) -> bool:
        mapping = self.associations()
        available = sorted(t for t in mapping if frame in self.tracks[t])
        record = {'frame': frame, 'pass': pass_id, 'correspondences': len(available), 'status': 'insufficient_support'}
        self.report['attempts'].append(record)
        if len(available) < 40:
            self.save_report()
            return False
        # Reserve observations on the first viable attempt and never allow them
        # into later retries, triangulation or BA.
        if frame not in self.holdouts:
            held = np.random.default_rng(frame).permutation(available)[::4].tolist()
            self.holdouts[frame] = held
            self.excluded.update((t, frame) for t in held)
        train = [t for t in available if (t, frame) not in self.excluded]
        if len(train) < 30:
            self.save_report()
            return False
        xyz = np.array([self.model.points[mapping[t]].xyz for t in train], dtype=np.float64)
        uv = np.array([self.tracks[t][frame] for t in train], dtype=np.float64)
        camera = self.calibration[frame]
        cv2.setRNGSeed(frame)
        ok, rotation, translation, inliers = cv2.solvePnPRansac(
            xyz, uv, camera.intrinsics, None, iterationsCount=2000,
            reprojectionError=3.0, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < 30:
            record['status'] = 'pnp_failed'
            self.save_report()
            return False
        selected = inliers.ravel()
        rotation, translation = cv2.solvePnPRefineLM(xyz[selected], uv[selected], camera.intrinsics,
                                                    None, rotation, translation)
        r, _ = cv2.Rodrigues(rotation)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = r.T, -r.T @ translation.ravel()
        candidate = replace(camera, pose=pose)
        good = [t for t in train if reprojection_error(candidate, self.model.points[mapping[t]].xyz, self.tracks[t][frame]) <= 3]
        h, w = camera.size_hw
        cells = {(min(3, max(0, int(self.tracks[t][frame][0]*4/w))),
                  min(3, max(0, int(self.tracks[t][frame][1]*4/h)))) for t in good}
        held = self.holdouts[frame]
        passed = sum(t in mapping and reprojection_error(candidate, self.model.points[mapping[t]].xyz,
                                                        self.tracks[t][frame]) <= 3 for t in held)
        record.update(training_inliers=len(good), training_fraction=len(good)/len(train),
                      held_out_count=len(held), held_out_fraction=passed/len(held), grid_cells=len(cells))
        if len(good) < 30 or len(good)/len(train) < 0.7 or passed/len(held) < 0.7 or len(cells) < 6:
            record['status'] = 'pose_validation_failed'
            self.save_report()
            return False
        points = list(self.model.points)
        for t in good:
            i = mapping[t]
            points[i] = replace(points[i], observations={**points[i].observations, frame: self.tracks[t][frame]})
        self.model = SparseModel({**self.model.cameras, frame: candidate}, tuple(points))
        record['status'] = 'registered'
        self.save_report()
        return True

    def extend_tracks(self) -> int:
        mapping = self.associations()
        points = list(self.model.points)
        added = 0
        claimed = self.claimed
        colors = {}
        for t, track in enumerate(self.tracks):
            observations = {f: uv for f, uv in track.items() if f in self.model.cameras and (t, f) not in self.excluded}
            if t in mapping:
                i = mapping[t]
                valid = {f: uv for f, uv in observations.items()
                         if reprojection_error(self.model.cameras[f], points[i].xyz, uv) <= 3
                         and (key(f, uv) not in claimed or
                              (f in points[i].observations and key(f, uv) == key(f, points[i].observations[f])))}
                points[i] = replace(points[i], observations={**points[i].observations, **valid})
                claimed.update(key(f, uv) for f, uv in valid.items())
                continue
            # Never create a second point for observations already owned by a
            # point whose global-track association is ambiguous.
            if any(key(f, uv) in claimed for f, uv in observations.items()):
                continue
            result = TrackTriangulator.triangulate(observations, self.model.cameras, max_error=3)
            if result is not None:
                xyz, valid = result
                frame, uv = next(iter(valid.items()))
                if frame not in colors:
                    with Image.open(self.run / 'processed' / f'{frame:06d}.png') as image:
                        colors[frame] = np.array(image.convert('RGB'))
                image = colors[frame]
                x, y = np.clip(np.rint(uv).astype(int), [0, 0], [image.shape[1]-1, image.shape[0]-1])
                points.append(SparsePoint(xyz, image[y, x].copy(), valid))
                claimed.update(key(f, uv) for f, uv in valid.items())
                added += 1
        self.model = SparseModel(self.model.cameras, tuple(points))
        return added

    @property
    def claimed(self) -> set[tuple]:
        return {key(f, uv) for p in self.model.points for f, uv in p.observations.items()}

    def validate(self) -> dict:
        mapping = self.associations()
        checks = []
        for frame in self.targets:
            if frame not in self.model.cameras:
                continue
            held = self.holdouts[frame]
            good = sum(t in mapping and reprojection_error(self.model.cameras[frame], self.model.points[mapping[t]].xyz,
                       self.tracks[t][frame]) <= 3 for t in held)
            checks.append({'frame': frame, 'held_out_count': len(held), 'within_3px': good/len(held)})
        return {'passed': all(c['within_3px'] >= 0.7 for c in checks), 'frames': checks}

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

    def save_report(self) -> None:
        self.report['registered_targets'] = [f for f in self.targets if f in self.model.cameras]
        self.report['missing_targets'] = [f for f in self.targets if f not in self.model.cameras]
        self.report['held_out_track_ids'] = {str(f): [self.track_ids[t] for t in ids]
                                           for f, ids in self.holdouts.items()}
        write_json(self.output / 'report.json', self.report)


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
