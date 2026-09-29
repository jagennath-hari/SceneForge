"""SceneForge-owned geometry and its coordinate/shape contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch

DepthUnits = Literal["reconstruction_units", "meters"]
ImageSize = tuple[int, int]  # height, width


def _floating_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")


@dataclass(frozen=True, slots=True)
class FrameGeometry:
    """Geometry on the processed RGB pixel grid.

    depth [H,W]: positive camera Z, not Euclidean ray distance.
    confidence [H,W]: raw model scores, not probabilities.
    intrinsics [3,3]: pinhole K in processed pixels.
    camera_to_world [4,4]: camera X right, Y down, Z forward (OpenCV).
    Depth and translations share the sequence's units. Invalid depth is retained
    for diagnostics and excluded by validity masks. Fields are frozen; tensors
    remain mutable, so consumers should treat their contents as read-only.
    """

    frame_index: int
    depth: torch.Tensor
    confidence: torch.Tensor
    intrinsics: torch.Tensor
    camera_to_world: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    @property
    def image_size_hw(self) -> ImageSize:
        return self.depth.shape[0], self.depth.shape[1]

    def validate(self) -> None:
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        for name in ("depth", "confidence", "intrinsics", "camera_to_world"):
            value = getattr(self, name)
            _floating_tensor(name, value)
            if value.device != self.depth.device:
                raise ValueError("A frame's geometry tensors must share a device")
        if self.depth.ndim != 2 or min(self.depth.shape) == 0:
            raise ValueError("depth must have nonempty shape [H,W]")
        if self.confidence.shape != self.depth.shape:
            raise ValueError("confidence must match the depth pixel grid")
        for name, shape in (("intrinsics", (3, 3)), ("camera_to_world", (4, 4))):
            value = getattr(self, name)
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
        k = self.intrinsics
        if k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError("Focal lengths must be positive")
        if not torch.allclose(k[2], k.new_tensor([0, 0, 1]), atol=1e-5, rtol=0):
            raise ValueError("K must have bottom row [0,0,1]")
        if not torch.isclose(k[1, 0], k.new_tensor(0), atol=1e-5, rtol=0):
            raise ValueError("Pinhole K must be upper triangular")
        pose = self.camera_to_world.float()
        if not torch.allclose(pose[3], pose.new_tensor([0, 0, 0, 1]), atol=1e-5, rtol=0):
            raise ValueError("camera_to_world must be homogeneous")
        rotation = pose[:3, :3]
        identity = torch.eye(3, device=pose.device, dtype=pose.dtype)
        if not torch.allclose(rotation.T @ rotation, identity, atol=1e-3, rtol=0):
            raise ValueError("Camera rotation must be orthonormal")
        if not torch.isclose(torch.linalg.det(rotation), pose.new_tensor(1), atol=1e-3, rtol=0):
            raise ValueError("Camera rotation must have determinant +1")


@dataclass(frozen=True, slots=True)
class GeometrySequence:
    """Geometry in one coordinate system, jointly inferred or aligned across sections."""

    frames: tuple[FrameGeometry, ...]
    processed_rgb: torch.Tensor  # [N,3,H,W], floating-point RGB in [0,1]
    source_names: tuple[str, ...]
    original_sizes_hw: tuple[ImageSize, ...]
    units: DepthUnits = "reconstruction_units"
    meters_per_unit: float | None = None

    def __post_init__(self) -> None:
        # Accept sequence inputs at runtime, but do not retain mutable lists.
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "source_names", tuple(self.source_names))
        object.__setattr__(self, "original_sizes_hw", tuple(tuple(s) for s in self.original_sizes_hw))
        self.validate()

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def image_size_hw(self) -> ImageSize:
        return self.frames[0].image_size_hw

    def validate(self) -> None:
        if not self.frames:
            raise ValueError("GeometrySequence must contain frames")
        for frame in self.frames:
            frame.validate()
        if not isinstance(self.units, str) or self.units not in {"meters", "reconstruction_units"}:
            raise ValueError("Unknown geometry units")
        if self.units == "meters":
            scale = self.meters_per_unit
            if isinstance(scale, bool) or not isinstance(scale, (int, float)):
                raise ValueError("Metric geometry requires an explicit numeric scale")
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("Metric geometry requires an explicit positive scale")
        elif self.meters_per_unit is not None:
            raise ValueError("Unscaled geometry must not declare a metric scale")
        h, w = self.image_size_hw
        _floating_tensor("processed_rgb", self.processed_rgb)
        if self.processed_rgb.shape != (len(self), 3, h, w):
            raise ValueError("RGB and geometry must use the same processed pixel grid")
        if not torch.isfinite(self.processed_rgb).all() or not (
            (self.processed_rgb >= 0).all() and (self.processed_rgb <= 1).all()
        ):
            raise ValueError("Processed RGB must be finite and in [0,1]")
        if len(self.source_names) != len(self) or len(self.original_sizes_hw) != len(self):
            raise ValueError("Source metadata must match the number of frames")
        if any(not isinstance(name, str) or not name for name in self.source_names):
            raise ValueError("Source names must be nonempty strings")
        if any(len(size) != 2 or any(type(v) is not int or v <= 0 for v in size)
               for size in self.original_sizes_hw):
            raise ValueError("Original dimensions must be positive integer (height,width) pairs")
        indices = [frame.frame_index for frame in self.frames]
        if indices != sorted(set(indices)):
            raise ValueError("Frame indices must be unique and increasing")
        if any(frame.image_size_hw != (h, w) for frame in self.frames):
            raise ValueError("All frames must share a processed resolution")
