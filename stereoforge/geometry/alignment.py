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
    pose_error_degrees: float = 0.0
    shared_frames: int = 0

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
                "inlier_fraction": self.inlier_fraction, "pose_error_degrees": self.pose_error_degrees,
                "shared_frames": self.shared_frames}


def _points(frame: FrameGeometry, y: torch.Tensor, x: torch.Tensor) -> np.ndarray:
    pixels = torch.stack((x, y, torch.ones_like(x))).double()
    camera = torch.linalg.solve(frame.intrinsics.double(), pixels) * frame.depth[y, x].double()
    pose = frame.camera_to_world.double()
    return (pose[:3, :3] @ camera + pose[:3, 3, None]).T.numpy()


def _fit(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Similarity fitting requires at least three finite point pairs")
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


class OverlapAlignmentError(ValueError):
    """Registration failed quality checks; retain numeric diagnostics."""

    def __init__(self, message: str, diagnostics: dict) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True, slots=True)
class _Candidate:
    name: str
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    errors: np.ndarray
    angles: np.ndarray

    @property
    def accepted(self) -> bool:
        return bool(np.median(self.errors) <= 0.08 and np.mean(self.errors <= 0.08) >= 0.5
                and np.median(self.angles) <= 10)

    @property
    def rank(self) -> tuple[bool, float, float, float]:
        return (self.accepted, float(np.mean(self.errors <= 0.08)),
                -float(np.median(self.errors)), -float(np.median(self.angles)))

    def summary(self) -> dict:
        return {"seed": self.name, "accepted": self.accepted, "scale": self.scale,
                "relative_error": float(np.median(self.errors)),
                "inlier_fraction": float(np.mean(self.errors <= 0.08)),
                "pose_error_degrees": float(np.median(self.angles))}


def align_overlap(reference: dict[int, FrameGeometry], local: GeometrySequence) -> SimilarityTransform:
    """Fit shared pixels using pose-guided seeds and deterministic Sim(3) RANSAC.

    The 8% depth-relative residual and 10-degree pose checks are initial quality
    gates, not calibrated guarantees. Failed registration stops publication.
    """
    source, target, depths, shared, scales = [], [], [], [], []
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
        scales.append(float(torch.median(other.depth[y, x].double() / frame.depth[y, x].double())))
    if len(shared) < 2:
        raise ValueError("Need at least two overlapping frames with reliable depth to merge sections")
    offsets = np.cumsum([0, *(len(points) for points in source)])
    source, target, depths = np.concatenate(source), np.concatenate(target), np.concatenate(depths)
    rotations_a = np.stack([a.camera_to_world[:3, :3].double().numpy() for a, _ in shared])
    rotations_b = np.stack([b.camera_to_world[:3, :3].double().numpy() for _, b in shared])
    candidates: list[_Candidate] = []

    def evaluate(name: str, fit: tuple[float, np.ndarray, np.ndarray]) -> _Candidate | None:
        scale, rotation, translation = fit
        if (not np.isfinite(scale) or scale <= 0 or not np.isfinite(rotation).all()
                or not np.isfinite(translation).all()):
            return None
        errors = np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1) / depths
        delta = (rotation @ rotations_a) @ rotations_b.transpose(0, 2, 1)
        angles = np.degrees(np.arccos(np.clip((np.trace(delta, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        if not np.isfinite(errors).all() or not np.isfinite(angles).all():
            return None
        return _Candidate(name, scale, rotation, translation, errors, angles)

    def propose(name: str, indices: np.ndarray) -> None:
        try:
            candidate = evaluate(name, _fit(source[indices], target[indices]))
        except ValueError:
            return
        if candidate is not None:
            candidates.append(candidate)

    propose("all_points", np.arange(len(source)))
    # A shared image supplies a camera rotation and a robust depth-scale ratio.
    # These provide independent seeds even when a global least-squares fit is bad.
    for i, (frame, other) in enumerate(shared):
        rotation = rotations_b[i] @ rotations_a[i].T
        translation = (other.camera_to_world[:3, 3].double().numpy() -
                       scales[i] * rotation @ frame.camera_to_world[:3, 3].double().numpy())
        candidate = evaluate(f"camera_{frame.frame_index}", (scales[i], rotation, translation))
        if candidate is not None:
            candidates.append(candidate)
        propose(f"frame_{frame.frame_index}", np.arange(offsets[i], offsets[i+1]))

    rng = np.random.default_rng(0)
    for attempt in range(256):
        # Spread hypotheses across views so one depth map cannot dominate every seed.
        if len(shared) >= 3:
            views = rng.choice(len(shared), 3, replace=False)
            indices = np.array([rng.integers(offsets[i], offsets[i+1]) for i in views])
        else:
            indices = rng.choice(len(source), 3, replace=False)
        propose(f"ransac_{attempt}", indices)
    if not candidates:
        raise OverlapAlignmentError("No nondegenerate overlap alignment hypothesis", {
            "shared_frame_indices": [a.frame_index for a, _ in shared], "point_count": len(source)})

    # Refit actual 8%-threshold inliers, rather than always retaining 70% even
    # when most of them disagree. Keep the original if refinement worsens it.
    candidates.sort(key=lambda candidate: candidate.rank, reverse=True)
    refined: list[_Candidate] = []
    for initial in candidates[:8]:
        current = initial
        for _ in range(6):
            selected = np.flatnonzero(current.errors <= 0.08)
            if len(selected) < 3:
                break
            try:
                candidate = evaluate(initial.name + "_refit", _fit(source[selected], target[selected]))
            except ValueError:
                break
            if candidate is None or candidate.rank <= current.rank:
                break
            current = candidate
        refined.append(current)
    best = max([*candidates, *refined], key=lambda candidate: candidate.rank)
    if not best.accepted:
        diagnostics = {"shared_frame_indices": [a.frame_index for a, _ in shared],
                       "point_count": len(source), "hypotheses": len(candidates),
                       "thresholds": {"relative_error": 0.08, "inlier_fraction": 0.5, "pose_degrees": 10},
                       "best": best.summary(),
                       "per_frame": [{"frame_index": frame.frame_index,
                                      "relative_error": float(np.median(best.errors[offsets[i]:offsets[i+1]])),
                                      "inlier_fraction": float(np.mean(best.errors[offsets[i]:offsets[i+1]] <= 0.08)),
                                      "pose_error_degrees": float(best.angles[i]),
                                      "depth_scale_ratio": scales[i]} for i, (frame, _) in enumerate(shared)]}
        raise OverlapAlignmentError(
            f"Unreliable overlap alignment: median relative error={np.median(best.errors):.3f}, "
            f"inliers={np.mean(best.errors <= 0.08):.1%}, "
            f"median pose disagreement={np.median(best.angles):.1f} degrees", diagnostics)
    return SimilarityTransform(best.scale, best.rotation, best.translation,
                               float(np.median(best.errors)), float(np.mean(best.errors <= 0.08)),
                               float(np.median(best.angles)), len(shared))
