"""Shared PnP registration, track extension and held-out validation for one map."""

from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from stereoforge.refinement.sparse_model import SparseCamera, SparseModel, SparsePoint, reprojection_error
from stereoforge.refinement.bundle_adjustment import SparseBundleAdjuster
from stereoforge.utils.artifacts import write_json
from .triangulation import TrackTriangulator


def key(frame: int, uv: np.ndarray) -> tuple:
    return frame, *map(float, np.round(uv, 4))


class MapRegistration:
    """Operations shared by bounded recovery and full-sequence reconstruction.

    Subclasses initialize the map, track identities, calibration and report state.
    """

    run: Path
    output: Path
    model: SparseModel
    ids: list[int]
    targets: list[int]
    tracks: list[dict[int, np.ndarray]]
    track_ids: list[int]
    lookup: dict[tuple, set[int]]
    calibration: dict[int, SparseCamera]
    holdouts: dict[int, list[int]]
    excluded: set[tuple[int, int]]
    optimizer: SparseBundleAdjuster
    package: Path
    report: dict

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

    def save_report(self) -> None:
        self.report['registered_targets'] = [f for f in self.targets if f in self.model.cameras]
        self.report['missing_targets'] = [f for f in self.targets if f not in self.model.cameras]
        self.report['held_out_track_ids'] = {str(f): [self.track_ids[t] for t in ids]
                                           for f, ids in self.holdouts.items()}
        write_json(self.output / 'report.json', self.report)

