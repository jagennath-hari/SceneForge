"""Robust overlap registration in a shared, non-metric reconstruction frame."""

from dataclasses import dataclass, replace

import numpy as np
import torch

from .geometry_utils import confidence_filtered_mask
from .types import FrameGeometry, GeometrySequence


@dataclass(frozen=True, slots=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    relative_error: float
    inlier_fraction: float

    def apply(self, frame: FrameGeometry) -> FrameGeometry:
        pose = frame.camera_to_world.clone()
        rotation = torch.as_tensor(self.rotation, dtype=pose.dtype)
        translation = torch.as_tensor(self.translation, dtype=pose.dtype)
        pose[:3, :3] = rotation @ pose[:3, :3]
        pose[:3, 3] = self.scale * (rotation @ pose[:3, 3]) + translation
        return replace(frame, depth=frame.depth * self.scale, camera_to_world=pose)

    def as_dict(self) -> dict:
        return {"scale": self.scale, "rotation": self.rotation.tolist(),
                "translation": self.translation.tolist(), "relative_error": self.relative_error,
                "inlier_fraction": self.inlier_fraction}


def _points(frame: FrameGeometry, y: torch.Tensor, x: torch.Tensor) -> np.ndarray:
    pixels = torch.stack((x, y, torch.ones_like(x))).double()
    camera = torch.linalg.solve(frame.intrinsics.double(), pixels) * frame.depth[y, x].double()
    pose = frame.camera_to_world.double()
    return (pose[:3, :3] @ camera + pose[:3, 3, None]).T.numpy()


def _fit(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    a, b = source.mean(axis=0), target.mean(axis=0)
    x, y = source - a, target - b
    try:
        u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Overlap similarity fit did not converge") from exc
    if singular[0] <= 0 or singular[1] < singular[0] * 1e-6:
        raise ValueError("Overlap geometry is degenerate; cannot determine a reliable alignment")
    sign = np.ones(3)
    sign[-1] = -1 if np.linalg.det(u @ vt) < 0 else 1
    rotation = (u * sign) @ vt
    scale = float((singular * sign).sum() / np.mean(np.sum(x * x, axis=1)))
    translation = b - scale * (rotation @ a)
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(translation).all():
        raise ValueError("Overlap produced an invalid similarity transform")
    return scale, rotation, translation


def align_overlap(reference: dict[int, FrameGeometry], local: GeometrySequence) -> SimilarityTransform:
    """Match the same pixels in shared frames; trim outliers before fitting Sim(3).

    The 8% depth-relative residual and 10-degree pose checks are initial quality
    gates, not calibrated guarantees. Failed registration stops publication.
    """
    source, target, depths, shared = [], [], [], []
    for frame in local.frames:
        other = reference.get(frame.frame_index)
        if other is None:
            continue
        if other.image_size_hw != frame.image_size_hw:
            raise ValueError("Overlap processed image grids differ")
        mask_a, _ = confidence_filtered_mask(frame.depth, frame.confidence, 50)
        mask_b, _ = confidence_filtered_mask(other.depth, other.confidence, 50)
        y, x = torch.where(mask_a & mask_b)
        if len(y) < 32:
            continue
        indices = torch.linspace(0, len(y) - 1, min(512, len(y))).long()
        y, x = y[indices], x[indices]
        source.append(_points(frame, y, x))
        target.append(_points(other, y, x))
        depths.append(other.depth[y, x].double().numpy())
        shared.append((frame, other))
    if len(shared) < 2:
        raise ValueError("Need at least two overlapping frames with reliable depth to merge sections")
    source, target, depths = np.concatenate(source), np.concatenate(target), np.concatenate(depths)
    selected = np.ones(len(source), dtype=bool)
    for _ in range(6):
        scale, rotation, translation = _fit(source[selected], target[selected])
        errors = np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1) / depths
        selected = errors <= np.quantile(errors, 0.7)
    scale, rotation, translation = _fit(source[selected], target[selected])
    errors = np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1) / depths
    relative_error = float(np.median(errors))
    inlier_fraction = float(np.mean(errors <= 0.08))
    angles = []
    for frame, other in shared:
        delta = (rotation @ frame.camera_to_world[:3, :3].double().numpy()) @ other.camera_to_world[:3, :3].double().numpy().T
        angles.append(np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))))
    if relative_error > 0.08 or inlier_fraction < 0.5 or np.median(angles) > 10:
        raise ValueError(
            f"Unreliable overlap alignment: median relative error={relative_error:.3f}, "
            f"inliers={inlier_fraction:.1%}, median pose disagreement={np.median(angles):.1f} degrees"
        )
    return SimilarityTransform(scale, rotation, translation, relative_error, inlier_fraction)
