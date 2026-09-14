"""Camera-convention conversion at third-party integration boundaries."""

from __future__ import annotations

import numpy as np
import torch


def invert_world_to_camera(extrinsics: torch.Tensor) -> torch.Tensor:
    """Invert [N,3,4] rigid extrinsics using R.T and -R.T @ t."""
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (3, 4):
        raise ValueError("Expected world-to-camera extrinsics [N,3,4]")
    if not extrinsics.is_floating_point() or not torch.isfinite(extrinsics).all():
        raise ValueError("Extrinsics must be finite floating-point tensors")
    source = extrinsics.float()
    rotation = source[:, :3, :3]
    identity = torch.eye(3, device=source.device).expand(len(source), 3, 3)
    if not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-3, rtol=0):
        raise ValueError("Extrinsics contain a non-rigid rotation")
    if not torch.allclose(torch.linalg.det(rotation), source.new_ones(len(source)), atol=1e-3, rtol=0):
        raise ValueError("Extrinsic rotations must have determinant +1")
    poses = torch.eye(4, device=source.device).repeat(len(source), 1, 1)
    poses[:, :3, :3] = rotation.transpose(-1, -2)
    poses[:, :3, 3] = -(rotation.transpose(-1, -2) @ source[:, :3, 3:]).squeeze(-1)
    return poses


def rotation_from_quaternion(values: list[float]) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid COLMAP quaternion")
    w, x, y, z = quaternion / norm
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                     [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])


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
