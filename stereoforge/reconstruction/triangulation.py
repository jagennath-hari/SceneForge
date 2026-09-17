"""Robust sparse initialization/retriangulation of fixed global image tracks."""

from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from stereoforge.refinement.sparse_model import SparseCamera, SparseModel, SparsePoint, reprojection_error
from stereoforge.utils.progress import tracked


class TrackTriangulator:
    def __init__(self, tracks: list[dict[int, np.ndarray]], images: dict[int, Path]) -> None:
        self.tracks, self.images = tracks, images

    @staticmethod
    def _fit(observations: dict, cameras: dict[int, SparseCamera]) -> np.ndarray | None:
        rows = []
        for frame, uv in observations.items():
            camera = cameras[frame]
            r = camera.pose[:3, :3].T
            projection = np.column_stack((r, -r @ camera.pose[:3, 3]))
            ray = np.linalg.solve(camera.intrinsics, [*uv, 1])
            rows.extend((ray[0]*projection[2]-projection[0], ray[1]*projection[2]-projection[1]))
        try:
            _, _, vt = np.linalg.svd(np.array(rows), full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        if abs(vt[-1, 3]) < 1e-10:
            return None
        xyz = vt[-1, :3] / vt[-1, 3]
        return xyz if np.isfinite(xyz).all() else None

    def build(self, cameras: dict[int, SparseCamera], max_error: float = 4.0) -> tuple[SparseModel, dict]:
        points, supports = [], {f: 0 for f in cameras}
        colors: dict[int, np.ndarray] = {}
        with tracked(self.tracks, "Triangulating global tracks", len(self.tracks), "track") as pending:
            for track in pending:
                observations = {f: uv for f, uv in track.items() if f in cameras}
                if len(observations) < 3:
                    continue
                ids = sorted(observations)
                sampled = [ids[i] for i in np.linspace(0, len(ids)-1, min(12, len(ids)), dtype=int)]
                rays = {}
                for f in sampled:
                    ray = cameras[f].pose[:3, :3] @ np.linalg.solve(cameras[f].intrinsics, [*observations[f], 1])
                    rays[f] = ray / np.linalg.norm(ray)
                pairs = []
                for a, b in combinations(sampled, 2):
                    angle = float(np.degrees(np.arccos(np.clip(rays[a] @ rays[b], -1, 1))))
                    if 1 <= angle <= 90:
                        pairs.append((angle, a, b))
                best, best_support, best_error = None, {}, float("inf")
                for _, a, b in sorted(pairs, reverse=True)[:16]:
                    xyz = self._fit({a: observations[a], b: observations[b]}, cameras)
                    if xyz is None:
                        continue
                    errors = {f: reprojection_error(cameras[f], xyz, uv) for f, uv in observations.items()}
                    valid = {f: observations[f] for f, error in errors.items() if error <= max_error}
                    error = float(np.median([errors[f] for f in valid])) if valid else float("inf")
                    if (len(valid), -error) > (len(best_support), -best_error):
                        best, best_support, best_error = xyz, valid, error
                if best is None or len(best_support) < 3:
                    continue
                xyz = self._fit(best_support, cameras)
                if xyz is None:
                    continue
                valid = {f: uv for f, uv in best_support.items() if reprojection_error(cameras[f], xyz, uv) <= max_error}
                if len(valid) < 3:
                    continue
                rgb = []
                for f, uv in valid.items():
                    if f not in colors:
                        with Image.open(self.images[f]) as image:
                            colors[f] = np.array(image.convert("RGB"))
                    h, w = colors[f].shape[:2]
                    x, y = np.clip(np.rint(uv).astype(int), [0, 0], [w-1, h-1])
                    rgb.append(colors[f][y, x])
                    supports[f] += 1
                points.append(SparsePoint(xyz, np.rint(np.mean(rgb, axis=0)).astype(np.uint8), valid))
        supported = {f: camera for f, camera in cameras.items() if supports[f]}
        report = {"candidate_tracks": len(self.tracks), "landmarks": len(points),
                  "observations_per_frame": supports, "unsupported_frames": sorted(cameras.keys()-supported.keys()),
                  "max_reprojection_pixels": max_error, "minimum_observations": 3,
                  "minimum_ray_angle_degrees": 1, "depth_prior": False}
        if not points:
            return SparseModel({}, ()), report
        return SparseModel(supported, tuple(points)), report
