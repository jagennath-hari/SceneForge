"""Sparse map snapshots exported by the native builder for output and dense refinement."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SparseCamera:
    """Pinhole camera in the map's arbitrary-scale world coordinate system."""

    pose: np.ndarray  # [4, 4] camera-to-world; optical axes right, down, forward
    intrinsics: np.ndarray  # [3, 3], in processed-image pixels
    size_hw: tuple[int, int]


@dataclass(frozen=True, slots=True)
class SparsePoint:
    """One landmark and its measured image observations."""

    xyz: np.ndarray  # [3], world coordinates
    rgb: np.ndarray  # [3], uint8 RGB
    observations: dict[int, np.ndarray]  # keyframe index -> [2] pixel coordinates


@dataclass(frozen=True, slots=True)
class SparseModel:
    """Camera and landmark snapshot; optimization state remains in C++."""

    cameras: dict[int, SparseCamera]
    points: tuple[SparsePoint, ...]
