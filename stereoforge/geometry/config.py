"""Validated configuration shared by the CLI and Python API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
import re
from typing import Any

import yaml

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


@dataclass(frozen=True, slots=True)
class PreviewConfig:
    confidence_percentile: float = 20.0
    max_points: int = 240000

    def __post_init__(self) -> None:
        require_finite("preview.confidence_percentile", self.confidence_percentile)
        if self.confidence_percentile > 100:
            raise ValueError("preview.confidence_percentile must be <= 100")
        require_integer("preview.max_points", self.max_points)


@dataclass(frozen=True, slots=True)
class RefinementConfig:
    enabled: bool = True
    backend: str = "pycusfm"
    feature_type: str = "aliked"
    device: int = 0
    debug: bool = False

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("refinement.enabled must be boolean")
        if type(self.debug) is not bool:
            raise ValueError("refinement.debug must be boolean")
        if self.backend != "pycusfm":
            raise ValueError("This refinement adapter requires pycusfm")
        if self.feature_type != "aliked":
            raise ValueError("refinement.feature_type must be aliked; this pipeline uses ALIKED exclusively")
        require_integer("refinement.device", self.device, minimum=0)


@dataclass(frozen=True, slots=True)
class DemoConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> DemoConfig:
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
        if document is None:
            document = {}
        if not isinstance(document, Mapping):
            raise ValueError("Configuration must be a YAML mapping")
        # Other pipeline sections are intentionally left to their own components.
        geometry = _section(document, "geometry", GeometryConfig)
        preview = _section(document, "preview", PreviewConfig)
        refinement = _section(document, "refinement", RefinementConfig)
        return cls(geometry=GeometryConfig(**geometry), preview=PreviewConfig(**preview),
                   refinement=RefinementConfig(**refinement))


def _section(document: Mapping[str, Any], name: str,
             model: type[GeometryConfig] | type[PreviewConfig] | type[RefinementConfig]) -> dict[str, Any]:
    section = document.get(name, {})
    if not isinstance(section, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    unknown = set(section) - {f.name for f in fields(model)}
    if unknown:
        raise ValueError(f"Unknown {name} settings: {', '.join(sorted(map(str, unknown)))}")
    return dict(section)
