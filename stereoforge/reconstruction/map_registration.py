"""Shared PnP registration, track extension and held-out validation for one map."""

from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

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

    refine_focal: bool = False
    validate_after_ba: bool = False

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
        if self.refine_focal:
            candidate, record['focal_refinement'] = self.refine_training_calibration(
                candidate, xyz, uv, selected, rotation, translation)
        good = [t for t in train if reprojection_error(candidate, self.model.points[mapping[t]].xyz, self.tracks[t][frame]) <= 3]
        h, w = camera.size_hw
        cells = {(min(3, max(0, int(self.tracks[t][frame][0]*4/w))),
                  min(3, max(0, int(self.tracks[t][frame][1]*4/h)))) for t in good}
        held = self.holdouts[frame]
        passed = sum(t in mapping and reprojection_error(candidate, self.model.points[mapping[t]].xyz,
                                                        self.tracks[t][frame]) <= 3 for t in held)
        record.update(training_inliers=len(good), training_fraction=len(good)/len(train),
                      held_out_count=len(held), held_out_fraction=passed/len(held), grid_cells=len(cells))
        training_passed = len(good) >= 30 and len(good)/len(train) >= 0.7 and len(cells) >= 6
        if not training_passed or (not self.validate_after_ba and passed/len(held) < 0.7):
            record['status'] = 'pose_validation_failed'
            self.save_report()
            return False
        points = list(self.model.points)
        for t in good:
            i = mapping[t]
            points[i] = replace(points[i], observations={**points[i].observations, frame: self.tracks[t][frame]})
        self.model = SparseModel({**self.model.cameras, frame: candidate}, tuple(points))
        record['status'] = 'provisional_for_ba' if self.validate_after_ba else 'registered'
        self.save_report()
        return True

    @staticmethod
    def refine_training_calibration(camera: SparseCamera, xyz: np.ndarray, uv: np.ndarray,
                                    selected: np.ndarray, rotation: np.ndarray,
                                    translation: np.ndarray) -> tuple[SparseCamera, dict]:
        """Fit pose and one focal multiplier; this API never receives holdouts.

        Use RANSAC training inliers for fitting and all training observations for
        model selection. Preserve principal point and fx/fy ratio. The +/-10%
        bound is a conservative experiment, not a general zoom calibration model.
        """
        initial_rotation = cv2.Rodrigues(rotation)[0]
        initial_translation = translation.ravel()
        camera_xyz = xyz @ initial_rotation.T + initial_translation
        positive = camera_xyz[:, 2] > 0
        details = {'status': 'insufficient_positive_depth', 'selected': False,
                   'initial_focal': camera.intrinsics.diagonal()[:2].tolist(),
                   'multiplier_bounds': [0.9, 1.1]}
        if not positive.any():
            return camera, details
        length = float(np.median(camera_xyz[positive, 2]))
        near = max(length * 1e-8, np.finfo(float).eps)

        def project(parameters: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            # Local increments and scene-normalized translation keep the solver
            # independent of the arbitrary reconstruction scale.
            delta_rotation = cv2.Rodrigues(parameters[:3])[0]
            transformed = (points @ initial_rotation.T + initial_translation) @ delta_rotation.T
            transformed += parameters[3:6] * length
            focal = camera.intrinsics.diagonal()[:2] * np.exp(parameters[6])
            pixels = transformed[:, :2] / np.maximum(transformed[:, 2:3], near)
            return pixels * focal + camera.intrinsics[:2, 2], transformed[:, 2]

        def residual(parameters: np.ndarray) -> np.ndarray:
            pixels, depth = project(parameters, xyz[selected])
            # Penalize crossing the camera plane as well as reprojection error.
            return np.concatenate(((pixels - uv[selected]).ravel(),
                                   np.minimum(depth - near, 0) / length * 1000))

        def score(parameters: np.ndarray) -> tuple[int, float]:
            pixels, depth = project(parameters, xyz)
            errors = np.linalg.norm(pixels - uv, axis=1)
            errors[depth <= near] = np.inf
            return int(np.count_nonzero(errors <= 3)), float(np.mean(np.minimum(errors, 3)**2))

        initial = np.zeros(7)
        lower, upper = np.full(7, -np.inf), np.full(7, np.inf)
        lower[6], upper[6] = np.log(0.9), np.log(1.1)
        result = least_squares(residual, initial, bounds=(lower, upper),
                               loss='soft_l1', f_scale=1.5, max_nfev=100, x_scale='jac')
        if not np.all(np.isfinite(result.x)) or not np.all(np.isfinite(result.jac)):
            details['status'] = 'nonfinite_solution'
            return camera, details
        multiplier = float(np.exp(result.x[6]))
        singular = np.linalg.svd(result.jac, compute_uv=False)
        ratio = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.0
        baseline_count, baseline_cost = score(initial)
        refined_count, refined_cost = score(result.x)
        at_bound = min(multiplier - 0.9, 1.1 - multiplier) < 0.001
        accepted = bool(result.success and not at_bound and ratio > 1e-8
                        and refined_count >= baseline_count and refined_cost < baseline_cost)
        details.update(status='selected' if accepted else 'kept_fixed_calibration',
                       selected=accepted, solver_success=bool(result.success),
                       solver_message=str(result.message), evaluations=int(result.nfev),
                       proposed_multiplier=multiplier, at_bound=at_bound,
                       jacobian_singular_ratio=ratio,
                       training_inliers_before=baseline_count, training_inliers_after=refined_count,
                       training_clipped_mse_before=baseline_cost, training_clipped_mse_after=refined_cost,
                       final_focal=(camera.intrinsics.diagonal()[:2] * (multiplier if accepted else 1)).tolist())
        if not accepted:
            return camera, details
        delta_rotation = cv2.Rodrigues(result.x[:3])[0]
        refined_rotation = delta_rotation @ initial_rotation
        refined_translation = delta_rotation @ initial_translation + result.x[3:6] * length
        pose = np.eye(4)
        pose[:3, :3] = refined_rotation.T
        pose[:3, 3] = -refined_rotation.T @ refined_translation
        intrinsics = camera.intrinsics.copy()
        intrinsics[0, 0] *= multiplier
        intrinsics[1, 1] *= multiplier
        return replace(camera, pose=pose, intrinsics=intrinsics), details

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

