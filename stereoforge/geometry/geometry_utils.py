"""Pure tensor functions for geometry filtering and depth statistics."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch


def _validate_maps(depth: torch.Tensor, confidence: torch.Tensor) -> None:
    if depth.ndim != 2 or depth.shape != confidence.shape:
        raise ValueError("Depth and confidence must have matching [H,W] shapes")
    if depth.device != confidence.device:
        raise ValueError("Depth and confidence must share a device")
    if not depth.is_floating_point() or not confidence.is_floating_point():
        raise ValueError("Depth and confidence must be floating point")


def make_valid_depth_mask(
    depth: torch.Tensor, confidence: torch.Tensor, confidence_threshold: float = 0.0,
) -> torch.Tensor:
    _validate_maps(depth, confidence)
    if not math.isfinite(confidence_threshold):
        raise ValueError("Confidence threshold must be finite")
    return (
        torch.isfinite(depth) & (depth > 0)
        & torch.isfinite(confidence) & (confidence >= confidence_threshold)
    )


def confidence_filtered_mask(
    depth: torch.Tensor, confidence: torch.Tensor, percentile: float = 20.0,
) -> tuple[torch.Tensor, float | None]:
    """Keep scores at or above the percentile; ties can retain more pixels."""
    if not math.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("Confidence percentile must be in [0,100]")
    valid = make_valid_depth_mask(depth, confidence)
    if not valid.any():
        return valid, None
    threshold = float(torch.quantile(confidence[valid].float(), percentile / 100))
    return valid & (confidence >= threshold), threshold


def depth_percentiles(
    depth: torch.Tensor, mask: torch.Tensor, percentiles: Sequence[float] = (10, 50, 90),
) -> dict[float, float | None]:
    """Return percentile values in input units; empty geometry produces None."""
    if depth.ndim != 2 or depth.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean and match depth [H,W]")
    if depth.device != mask.device or not depth.is_floating_point():
        raise ValueError("Depth must be floating point and share the mask's device")
    if any(not math.isfinite(p) or not 0 <= p <= 100 for p in percentiles):
        raise ValueError("Percentiles must be in [0,100]")
    if not percentiles:
        return {}
    values = depth[mask & torch.isfinite(depth) & (depth > 0)].float()
    if values.numel() == 0:
        return {p: None for p in percentiles}
    q = torch.tensor(percentiles, device=values.device, dtype=values.dtype) / 100
    return dict(zip(percentiles, torch.quantile(values, q).tolist(), strict=True))


def robust_near_depth(
    depth: torch.Tensor, confidence: torch.Tensor, percentile: float = 10.0,
    confidence_threshold: float = 0.0,
) -> float | None:
    mask = make_valid_depth_mask(depth, confidence, confidence_threshold)
    return depth_percentiles(depth, mask, (percentile,))[percentile]
