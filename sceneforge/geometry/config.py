"""Validated settings for the VGGT-Ω inference adapter."""

from __future__ import annotations

from dataclasses import dataclass
import re

from sceneforge.utils.validation import require_finite, require_integer


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    device: str = "cuda"
    image_resolution: int = 512
    preprocess_mode: str = "balanced"
    max_frames: int | None = None
    meters_per_unit: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or re.fullmatch(r"cpu|cuda(?::\d+)?", self.device) is None:
            raise ValueError("geometry.device must be cpu, cuda or cuda:N")
        require_integer("geometry.image_resolution", self.image_resolution)
        if self.image_resolution % 16:
            raise ValueError("geometry.image_resolution must be divisible by 16")
        if not isinstance(self.preprocess_mode, str) or self.preprocess_mode not in {"balanced", "max_size"}:
            raise ValueError("geometry.preprocess_mode must be balanced or max_size")
        if self.max_frames is not None:
            require_integer("geometry.max_frames", self.max_frames)
        if self.meters_per_unit is not None:
            require_finite("geometry.meters_per_unit", self.meters_per_unit)
            if self.meters_per_unit == 0:
                raise ValueError("geometry.meters_per_unit must be positive")
