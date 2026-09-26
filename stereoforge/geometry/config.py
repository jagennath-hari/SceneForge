"""Validated settings for the VGGT inference adapter."""

from __future__ import annotations

from dataclasses import dataclass
import re

from stereoforge.utils.validation import require_finite, require_integer


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    device: str = "cuda"
    image_resolution: int = 512
    preprocess_mode: str = "balanced"
    max_frames: int | None = None
    meters_per_unit: float | None = None
    chunk_max_frames: int | None = None
    chunk_overlap: int = 32
    gpu_memory_fraction: float = 0.90

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or re.fullmatch(r"cpu|cuda(?::\d+)?", self.device) is None:
            raise ValueError("geometry.device must be cpu, cuda or cuda:N")
        require_integer("geometry.chunk_overlap", self.chunk_overlap, minimum=5)
        if self.chunk_max_frames is not None:
            require_integer("geometry.chunk_max_frames", self.chunk_max_frames)
            if self.chunk_max_frames < self.chunk_overlap + 2:
                raise ValueError("chunk_max_frames must be at least chunk_overlap + 2")
        require_finite("geometry.gpu_memory_fraction", self.gpu_memory_fraction)
        if not 0 < self.gpu_memory_fraction <= 1:
            raise ValueError("gpu_memory_fraction must be in (0,1]")
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
