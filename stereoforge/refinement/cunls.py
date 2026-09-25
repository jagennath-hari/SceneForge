"""Thin file/process boundary to StereoForge's native cuNLS bundle adjuster."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import shutil
import signal
import subprocess

import numpy as np

from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import Progress
from .sparse_model import SparseCamera, SparseModel, SparsePoint, reprojection_error


@dataclass(frozen=True, slots=True)
class CuNLSOptions:
    threshold_pixels: float = 3.0
    focal_sigma_pixels: float = 10.0
    rotation_sigma_radians: float = 0.1
    translation_sigma: float = 0.1  # Units after median camera-step normalization.
    gnc_rounds: int = 64
    lm_iterations: int = 50
    check_jacobians: bool = False
    use_gnc: bool = True
    huber_delta_pixels: float = 0.0


class CuNLSBundleAdjuster:
    """Run one anchored local BA. No pyCuSFM or GTSfM imports or binaries."""

    def __init__(self, device: int = 0, options: CuNLSOptions | None = None) -> None:
        self.device = device
        self.options = options or CuNLSOptions()

    @staticmethod
    def preflight() -> str:
        executable = shutil.which('stereoforge-bundle-adjust')
        if executable is None:
            raise RuntimeError('Rebuild Docker to install stereoforge-bundle-adjust')
        return executable

    def optimize(self, model: SparseModel, directory: Path) -> tuple[SparseModel, dict]:
        ids = sorted(model.cameras)
        if len(ids) < 3 or not model.points:
            raise ValueError('Local BA requires at least three cameras and supported landmarks')
        centers = np.array([model.cameras[f].pose[:3, 3] for f in ids])
        steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        positive = steps[steps > 1e-8]
        if not len(positive):
            raise ValueError('Camera baseline is degenerate; monocular scale is unconstrained')
        scale = float(np.median(positive))
        origin = centers[0].copy()
        order = {frame: i for i, frame in enumerate(ids)}
        cameras = []
        for frame in ids:
            camera = model.cameras[frame]
            pose = camera.pose.copy()
            pose[:3, 3] = (pose[:3, 3] - origin) / scale
            k = camera.intrinsics
            cameras.append({'world_to_camera': np.linalg.inv(pose).ravel().tolist(),
                            'intrinsics': [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])]})
        request = {'format_version': 1, 'cameras': cameras,
                   'points': [((p.xyz - origin) / scale).tolist() for p in model.points],
                   'observations': [{'camera': order[f], 'point': i, 'pixel': uv.tolist()}
                                    for i, p in enumerate(model.points) for f, uv in sorted(p.observations.items())],
                   'options': asdict(self.options)}
        normalization = {'origin': origin.tolist(), 'scale': scale,
                         'frame_order': ids, 'units': 'reconstruction_units'}
        return self._solve(model, directory, request, normalization)

    def retry_joint(self, pair: Path, directory: Path, iterations: int) -> tuple[SparseModel, dict, SparseModel]:
        """Restart the original joint objective, changing only its iteration budget."""
        request = json.loads((pair / 'joint_ba/input.json').read_text())
        normalization = json.loads((pair / 'joint_ba/normalization.json').read_text())
        if request.get('format_version') != 1 or request['options'].get('use_gnc') is not False:
            raise ValueError('Expected a saved unit-weight joint BA input')
        if not request['options']['lm_iterations'] < iterations <= 500:
            raise ValueError('New iteration budget must exceed the saved budget and be at most 500')
        ids = normalization['frame_order']
        if ids != sorted(set(ids)) or len(ids) != len(request['cameras']):
            raise ValueError('Invalid saved camera identities')
        seed = SparseModel.read(pair / 'combined_sparse', ids)
        if set(seed.cameras) != set(ids) or len(seed.points) != len(request['points']):
            raise ValueError('Combined model does not match saved joint BA identities')
        origin = np.asarray(normalization['origin'], dtype=float)
        scale = float(normalization['scale'])
        if origin.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid saved normalization')
        cameras = {}
        for frame, record in zip(ids, request['cameras'], strict=True):
            pose = np.linalg.inv(np.asarray(record['world_to_camera'], dtype=float).reshape(4, 4))
            pose[:3, 3] = pose[:3, 3]*scale + origin
            fx, fy, cx, cy = record['intrinsics']
            cameras[frame] = SparseCamera(pose, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                                          seed.cameras[frame].size_hw)
        observations: list[dict] = [{} for _ in request['points']]
        for observation in request['observations']:
            camera, point = observation['camera'], observation['point']
            if not 0 <= camera < len(ids) or not 0 <= point < len(observations):
                raise ValueError('Unknown saved observation identity')
            frame, uv = ids[camera], np.asarray(observation['pixel'], dtype=float)
            if frame in observations[point] or frame not in seed.points[point].observations:
                raise ValueError('Duplicate or unknown saved observation')
            if not np.array_equal(uv, seed.points[point].observations[frame]):
                raise ValueError('Saved training pixels differ from combined model')
            observations[point][frame] = uv
        points = tuple(SparsePoint(np.asarray(xyz)*scale + origin, seed.points[i].rgb, observations[i])
                       for i, xyz in enumerate(request['points']))
        training = SparseModel(cameras, points)
        request['options']['lm_iterations'] = iterations
        optimizer = CuNLSBundleAdjuster(self.device, CuNLSOptions(**request['options']))
        refined, report = optimizer._solve(training, directory, request, normalization)
        return refined, report, training

    def _solve(self, model: SparseModel, directory: Path, request: dict,
               normalization: dict) -> tuple[SparseModel, dict]:
        executable = self.preflight()
        directory.mkdir()
        ids = normalization['frame_order']
        origin = np.asarray(normalization['origin'], dtype=float)
        scale = float(normalization['scale'])
        write_json(directory / 'input.json', request)
        write_json(directory / 'normalization.json', normalization)
        command = [executable, str(directory / 'input.json'), str(directory / 'output.json'), str(self.device)]
        with Progress('cuNLS local BA' if self.options.use_gnc else 'cuNLS joint BA'), (directory / 'solver.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait()
            except BaseException:
                try:
                    process.send_signal(signal.SIGTERM)
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                except ProcessLookupError:
                    pass
                raise
        if code:
            raise RuntimeError(f'cuNLS BA failed (exit {code}); inspect {directory / "solver.log"}')
        result = json.loads((directory / 'output.json').read_text())
        if self.options.huber_delta_pixels > 0 and result.get('report', {}).get('huber_delta_pixels') != self.options.huber_delta_pixels:
            raise RuntimeError('Native solver lacks the requested Huber loss; rebuild Docker')
        if not self.options.use_gnc and result.get('report', {}).get('use_gnc') is not False:
            raise RuntimeError('Native solver lacks joint-BA mode; rebuild Docker before running the pair diagnostic')
        if (result.get('format_version') != 1 or len(result['cameras']) != len(ids)
                or len(result['points']) != len(model.points)):
            raise ValueError('Native BA changed camera/point identities')
        optimized = {}
        for frame, record in zip(ids, result['cameras'], strict=True):
            world_to_camera = np.asarray(record['world_to_camera'], dtype=float).reshape(4, 4)
            pose = np.linalg.inv(world_to_camera)
            pose[:3, 3] = pose[:3, 3]*scale + origin
            fx, fy, cx, cy = record['intrinsics']
            k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
            if not np.isfinite(pose).all() or not np.isfinite(k).all() or min(fx, fy) <= 0:
                raise ValueError('Nonfinite or invalid native camera')
            optimized[frame] = SparseCamera(pose, k, model.cameras[frame].size_hw)
        raw_points = []
        retained = []
        retained_indices = []
        for index, (point, xyz) in enumerate(zip(model.points, result['points'], strict=True)):
            position = np.asarray(xyz, dtype=float)*scale + origin
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError('Invalid native landmark')
            raw_points.append(replace(point, xyz=position))
            valid = {f: uv for f, uv in point.observations.items()
                     if reprojection_error(optimized[f], position, uv) <= self.options.threshold_pixels}
            if len(valid) >= 3:
                retained.append(SparsePoint(position, point.rgb, valid))
                retained_indices.append(index)
        # Preserve raw results separately; filtering must not hide solver failures.
        SparseModel(optimized, tuple(raw_points)).write(directory / 'raw', ids)
        supported = {f for p in retained for f in p.observations}
        filtered = SparseModel({f: optimized[f] for f in sorted(supported)}, tuple(retained))
        filtered.write(directory / 'sparse', ids)
        support = {f: sum(f in p.observations for p in retained) for f in supported}
        adjacency = {f: set() for f in supported}
        for point in retained:
            for f in point.observations:
                adjacency[f].update(point.observations)
        reached = set()
        pending = [min(supported)] if supported else []
        while pending:
            f = pending.pop()
            if f not in reached:
                reached.add(f)
                pending.extend(adjacency[f] - reached)
        connected = bool(supported) and reached == supported
        adequate_support = all(support.get(f, 0) >= 6 for f in ids)
        complete = result['report'].get('optimization_complete', result['report']['gnc_converged'])
        report = {**result['report'], 'input_frames': len(ids), 'registered_frames': len(supported),
                  'missing_frames': sorted(set(ids)-supported), 'landmarks_before': len(model.points),
                  'landmarks_after': len(retained), 'connected': connected, 'camera_support': support,
                  'options': asdict(self.options), 'retained_point_indices': retained_indices,
                  'validation_scope': 'Optimization residuals, not independent accuracy or held-out validation',
                  'normalization_scale': scale, 'frame_order': ids,
                  'status': 'diagnostic_complete' if adequate_support and connected and complete
                            else 'diagnostic_partial'}
        write_json(directory / 'report.json', report)
        return filtered, report
