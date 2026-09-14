"""Initialize observed sparse landmarks from VGGT depth and serialize COLMAP."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import tracked
from .cusfm_tracks import NativeMatchesReader, Observation
from .types import GeometrySequence


def quaternion_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Return COLMAP's w,x,y,z quaternion for a world-to-camera rotation."""
    r = rotation
    matrix = np.array([
        [r[0, 0]-r[1, 1]-r[2, 2], r[1, 0]+r[0, 1], r[2, 0]+r[0, 2], r[2, 1]-r[1, 2]],
        [r[1, 0]+r[0, 1], r[1, 1]-r[0, 0]-r[2, 2], r[2, 1]+r[1, 2], r[0, 2]-r[2, 0]],
        [r[2, 0]+r[0, 2], r[2, 1]+r[1, 2], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]],
        [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)],
    ]) / 3
    _, vectors = np.linalg.eigh(matrix)
    quaternion = vectors[:, -1][[3, 0, 1, 2]]
    return quaternion if quaternion[0] >= 0 else -quaternion


@dataclass(frozen=True, slots=True)
class Landmark:
    xyz: np.ndarray
    rgb: np.ndarray
    observations: tuple[Observation, ...]
    error: float


class DepthLandmarkInitializer:
    """Use measured tracks, never projected synthetic correspondences, for BA."""

    def __init__(self, max_reprojection_error: float = 14.0) -> None:
        self.max_reprojection_error = max_reprojection_error

    def build(self, sequence: GeometrySequence, directory: Path, export: dict) -> dict:
        reader = NativeMatchesReader()
        keypoints = reader.keypoints(directory / "workspace/keyframes", [f.image_size_hw for f in sequence.frames])
        tracks, summary = reader.tracks(directory / "workspace/matches", keypoints)
        poses = np.stack([f.camera_to_world.numpy() for f in sequence.frames]).astype(np.float64)
        poses = np.array(export["initial_world_to_first_camera"]) @ poses
        factor = export["working_units_per_input_unit"]
        poses[:, :3, 3] *= factor
        intrinsics = np.stack([f.intrinsics.numpy() for f in sequence.frames]).astype(np.float64)
        world_to_camera = np.linalg.inv(poses)
        inverse_k = np.linalg.inv(intrinsics)
        landmarks = []
        with tracked(tracks, "Initializing VGGT landmarks", len(tracks), "track") as pending:
            for track in pending:
                positions, weights, observations, colors = [], [], [], []
                for observation in track:
                    index = observation.frame
                    u, v = keypoints[index][observation.keypoint]
                    height, width = sequence.frames[index].image_size_hw
                    x, y = min(width-1, int(round(u))), min(height-1, int(round(v)))
                    depth = float(sequence.frames[index].depth[y, x])
                    confidence = float(sequence.frames[index].confidence[y, x])
                    if not np.isfinite(depth) or depth <= 0 or not np.isfinite(confidence) or confidence <= 0:
                        continue
                    local = (inverse_k[index] @ [u, v, 1]) * depth * factor
                    positions.append(poses[index, :3, :3] @ local + poses[index, :3, 3])
                    weights.append(confidence)
                    observations.append(observation)
                    colors.append(sequence.processed_rgb[index, :, y, x].numpy())
                if len(observations) < 3:
                    continue
                weights = np.asarray(weights, dtype=np.float64)
                weights /= weights.sum()
                xyz = np.average(positions, axis=0, weights=weights)
                # Remove observations inconsistent with the initialized geometry,
                # then re-estimate from the surviving depth samples.
                keep = self._inliers(xyz, observations, keypoints, intrinsics, world_to_camera)
                if len(keep) < 3:
                    continue
                xyz = np.average(np.asarray(positions)[keep], axis=0, weights=weights[keep])
                observations = [observations[i] for i in keep]
                valid = self._inliers(xyz, observations, keypoints, intrinsics, world_to_camera)
                if len(valid) < 3:
                    continue
                observations = tuple(observations[i] for i in valid)
                errors = [self._error(xyz, o, keypoints, intrinsics, world_to_camera) for o in observations]
                rgb = np.clip(np.average(np.asarray(colors)[keep], axis=0, weights=weights[keep])*255, 0, 255).round()
                landmarks.append(Landmark(xyz, rgb.astype(np.uint8), observations, float(np.mean(errors))))
        if not landmarks:
            raise ValueError("No consistent depth-initialized tracks survived; inspect VGGT depth, poses and ALIKED tracks")
        initial = directory / "initialized_sparse"
        self._write_colmap(initial, sequence, keypoints, world_to_camera, intrinsics, landmarks)
        supported = {o.frame for point in landmarks for o in point.observations}
        summary.update(initialized_landmarks=len(landmarks), initialized_frames=len(supported),
                       unsupported_frame_indices=sorted(set(range(len(sequence))) - supported),
                       pre_ba_max_reprojection_error=self.max_reprojection_error,
                       min_track_length=3, initialization="confidence_weighted_vggt_depth",
                       depth_used_as="initialization only, not a persistent depth prior")
        write_json(directory / "initialization.json", summary)
        return summary

    @staticmethod
    def _error(xyz: np.ndarray, observation: Observation, keypoints: dict[int, np.ndarray],
               intrinsics: np.ndarray, world_to_camera: np.ndarray) -> float:
        i = observation.frame
        camera = world_to_camera[i, :3, :3] @ xyz + world_to_camera[i, :3, 3]
        if not np.isfinite(camera).all() or camera[2] <= 0:
            return float("inf")
        pixel = intrinsics[i] @ camera
        return float(np.linalg.norm(pixel[:2]/pixel[2] - keypoints[i][observation.keypoint]))

    def _inliers(self, xyz: np.ndarray, observations: list[Observation], keypoints: dict[int, np.ndarray],
                 intrinsics: np.ndarray, world_to_camera: np.ndarray) -> list[int]:
        return [i for i, observation in enumerate(observations)
                if self._error(xyz, observation, keypoints, intrinsics, world_to_camera) <= self.max_reprojection_error]

    @staticmethod
    def _write_colmap(directory: Path, sequence: GeometrySequence, keypoints: dict[int, np.ndarray],
                      world_to_camera: np.ndarray, intrinsics: np.ndarray, landmarks: list[Landmark]) -> None:
        directory.mkdir()
        frame_observations: dict[int, list[tuple[Observation, int]]] = {}
        point_tracks: dict[int, list[tuple[int, int]]] = {}
        for point_id, point in enumerate(landmarks, 1):
            point_tracks[point_id] = []
            for observation in point.observations:
                records = frame_observations.setdefault(observation.frame, [])
                point_tracks[point_id].append((observation.frame + 1, len(records)))
                records.append((observation, point_id))
        with (directory / "cameras.txt").open("w") as cameras, (directory / "images.txt").open("w") as images:
            for frame, observations in sorted(frame_observations.items()):
                identifier = frame + 1
                height, width = sequence.frames[frame].image_size_hw
                k = intrinsics[frame]
                cameras.write(f"{identifier} PINHOLE {width} {height} {k[0,0]:.17g} {k[1,1]:.17g} {k[0,2]:.17g} {k[1,2]:.17g}\n")
                pose = world_to_camera[frame]
                values = [*quaternion_from_rotation(pose[:3, :3]), *pose[:3, 3]]
                images.write(f"{identifier} " + " ".join(f"{v:.17g}" for v in values)
                             + f" {identifier} frame_{frame:06d}.png\n")
                images.write(" ".join(f"{keypoints[frame][o.keypoint,0]:.17g} {keypoints[frame][o.keypoint,1]:.17g} {pid}"
                                      for o, pid in observations) + "\n")
        with (directory / "points3D.txt").open("w") as points:
            for point_id, point in enumerate(landmarks, 1):
                points.write(f"{point_id} " + " ".join(f"{v:.17g}" for v in point.xyz)
                             + " " + " ".join(str(int(v)) for v in point.rgb) + f" {point.error:.17g} "
                             + " ".join(f"{frame} {index}" for frame, index in point_tracks[point_id]) + "\n")
