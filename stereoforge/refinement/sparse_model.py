"""Small, explicit COLMAP model for joining locally optimized sections."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from stereoforge.utils.camera import quaternion_from_rotation, rotation_from_quaternion


@dataclass(frozen=True, slots=True)
class SparseCamera:
    pose: np.ndarray  # camera-to-world
    intrinsics: np.ndarray
    size_hw: tuple[int, int]


@dataclass(frozen=True, slots=True)
class SparsePoint:
    xyz: np.ndarray
    rgb: np.ndarray
    observations: dict[int, np.ndarray]  # global frame index -> measured pixel


@dataclass(frozen=True, slots=True)
class SparseModel:
    cameras: dict[int, SparseCamera]
    points: tuple[SparsePoint, ...]

    @classmethod
    def read(cls, directory: Path, frame_indices: list[int]) -> "SparseModel":
        calibration = {}
        for line in (directory / "cameras.txt").read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 8 or fields[1] != "PINHOLE":
                raise ValueError("Section BA requires PINHOLE cameras")
            fx, fy, cx, cy = map(float, fields[4:])
            calibration[int(fields[0])] = (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                                           (int(fields[3]), int(fields[2])))
        cameras, observations = {}, {}
        with (directory / "images.txt").open() as stream:
            while line := stream.readline():
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) != 10 or not fields[9].startswith("frame_"):
                    raise ValueError("Malformed sparse image record")
                index = int(Path(fields[9]).stem.removeprefix("frame_"))
                if not 0 <= index < len(frame_indices):
                    raise ValueError("Sparse image refers to an unknown section frame")
                frame = frame_indices[index]
                if frame in cameras:
                    raise ValueError("Duplicate frame in sparse reconstruction")
                rotation = rotation_from_quaternion(list(map(float, fields[1:5])))
                pose = np.eye(4)
                pose[:3, :3] = rotation.T
                pose[:3, 3] = -rotation.T @ np.array(list(map(float, fields[5:8])))
                k, size = calibration[int(fields[8])]
                cameras[frame] = SparseCamera(pose, k, size)
                values = stream.readline().split()
                observations[int(fields[0])] = (frame, np.array(
                    [(float(values[i]), float(values[i+1])) for i in range(0, len(values), 3)]))
        points = []
        for line in (directory / "points3D.txt").read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            track = {}
            for i in range(8, len(fields), 2):
                frame, pixels = observations[int(fields[i])]
                if frame in track:
                    raise ValueError("Multiple observations of a landmark in one frame")
                track[frame] = pixels[int(fields[i+1])]
            points.append(SparsePoint(np.array(list(map(float, fields[1:4]))),
                                      np.array(list(map(int, fields[4:7]))), track))
        return cls(cameras, tuple(points))

    def transformed(self, scale: float, rotation: np.ndarray, translation: np.ndarray) -> "SparseModel":
        cameras = {}
        for frame, camera in self.cameras.items():
            pose = camera.pose.copy()
            pose[:3, :3] = rotation @ pose[:3, :3]
            pose[:3, 3] = scale * rotation @ pose[:3, 3] + translation
            cameras[frame] = replace(camera, pose=pose)
        points = tuple(replace(p, xyz=scale * rotation @ p.xyz + translation) for p in self.points)
        return SparseModel(cameras, points)

    def write(self, directory: Path, frame_indices: list[int]) -> None:
        directory.mkdir()
        order = {frame: i for i, frame in enumerate(frame_indices)}
        observations: dict[int, list[tuple[np.ndarray, int]]] = {}
        tracks = []
        for identifier, point in enumerate(self.points, 1):
            track = []
            for frame, uv in sorted(point.observations.items()):
                records = observations.setdefault(frame, [])
                track.append((order[frame] + 1, len(records)))
                records.append((uv, identifier))
            tracks.append(track)
        with (directory / "cameras.txt").open("w") as cs, (directory / "images.txt").open("w") as ims:
            for frame, records in sorted(observations.items()):
                camera = self.cameras[frame]
                identifier = order[frame] + 1
                h, w = camera.size_hw
                k = camera.intrinsics
                cs.write(f"{identifier} PINHOLE {w} {h} {k[0,0]:.17g} {k[1,1]:.17g} {k[0,2]:.17g} {k[1,2]:.17g}\n")
                r = camera.pose[:3, :3].T
                values = [*quaternion_from_rotation(r), *(-r @ camera.pose[:3, 3])]
                ims.write(f"{identifier} " + " ".join(f"{v:.17g}" for v in values)
                          + f" {identifier} frame_{order[frame]:06d}.png\n")
                ims.write(" ".join(f"{uv[0]:.17g} {uv[1]:.17g} {pid}" for uv, pid in records) + "\n")
        with (directory / "points3D.txt").open("w") as stream:
            for identifier, (point, track) in enumerate(zip(self.points, tracks, strict=True), 1):
                stream.write(f"{identifier} " + " ".join(f"{v:.17g}" for v in point.xyz)
                             + " " + " ".join(str(int(v)) for v in point.rgb) + " 0 "
                             + " ".join(f"{i} {j}" for i, j in track) + "\n")


def reprojection_error(camera: SparseCamera, xyz: np.ndarray, uv: np.ndarray) -> float:
    point = camera.pose[:3, :3].T @ (xyz - camera.pose[:3, 3])
    if not np.isfinite(point).all() or point[2] <= 0:
        return float("inf")
    pixel = camera.intrinsics @ point
    return float(np.linalg.norm(pixel[:2] / pixel[2] - uv))
