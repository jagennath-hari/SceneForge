"""Report serialization and previews, separate from inference and the CLI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
import torch

from stereoforge.geometry.config import PreviewConfig
from stereoforge.geometry.geometry_utils import confidence_filtered_mask, depth_percentiles
from stereoforge.geometry.types import GeometrySequence
from .artifacts import write_json
from .progress import Progress, tracked

FloatArray = NDArray[np.float32]
ByteArray = NDArray[np.uint8]
BoolArray = NDArray[np.bool_]


def _numpy(tensor: torch.Tensor) -> FloatArray:
    return tensor.detach().float().cpu().numpy()


def _display_range(values: FloatArray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, [2, 98])
    return float(low), float(high)


def _colorize(values: FloatArray, valid: BoolArray, low: float, high: float) -> ByteArray:
    """Yellow=near/high confidence, purple=far/low confidence; invalid=black."""
    # Float64 avoids overflow while normalizing very large but finite predictions.
    values64 = values.astype(np.float64)
    normalized = np.clip((values64 - low) / max(high - low, 1e-8), 0, 1)
    normalized = np.nan_to_num(normalized, nan=0, posinf=1, neginf=0)
    palette = np.array([[250, 225, 65], [60, 190, 130], [40, 110, 180], [65, 25, 95]])
    positions = np.linspace(0, 1, len(palette))
    rgb = np.stack([np.interp(normalized, positions, palette[:, c]) for c in range(3)], axis=-1)
    rgb[~valid] = 0
    return rgb.astype(np.uint8)


@dataclass(frozen=True, slots=True)
class GeometryArrays:
    depth: FloatArray
    confidence: FloatArray
    valid: BoolArray
    intrinsics: FloatArray
    poses: FloatArray
    rgb: ByteArray
    thresholds: tuple[float | None, ...]

    @classmethod
    def from_sequence(cls, sequence: GeometrySequence, percentile: float) -> GeometryArrays:
        pairs = [confidence_filtered_mask(f.depth, f.confidence, percentile) for f in sequence.frames]
        return cls(
            depth=np.stack([_numpy(f.depth) for f in sequence.frames]),
            confidence=np.stack([_numpy(f.confidence) for f in sequence.frames]),
            valid=np.stack([mask.detach().cpu().numpy() for mask, _ in pairs]),
            intrinsics=np.stack([_numpy(f.intrinsics) for f in sequence.frames]),
            poses=np.stack([_numpy(f.camera_to_world) for f in sequence.frames]),
            rgb=(_numpy(sequence.processed_rgb).transpose(0, 2, 3, 1) * 255).round().astype(np.uint8),
            thresholds=tuple(threshold for _, threshold in pairs),
        )


class GeometryReportWriter:
    """Write the existing NPZ/PNG/PLY/HTML format with bounded point sampling."""

    def __init__(self, config: PreviewConfig | None = None) -> None:
        self.config = config if config is not None else PreviewConfig()

    def write(
        self, sequence: GeometrySequence, output: Path, *,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        sequence.validate()
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        artifacts = ("geometry.npz", "metadata.json", "index.html", "previews", "contact_sheet.jpg", "point_cloud.ply")
        if any((output / name).exists() for name in artifacts):
            raise ValueError(f"Report artifacts already exist in {output}")
        with Progress("Preparing report arrays"):
            arrays = GeometryArrays.from_sequence(sequence, self.config.confidence_percentile)
            depth_range = _display_range(arrays.depth[arrays.valid])
            confidence_range = _display_range(arrays.confidence[np.isfinite(arrays.confidence)])
        with Progress("Compressing geometry.npz"):
            self._write_arrays(sequence, arrays, output)
        records = self._write_previews(sequence, arrays, output, depth_range, confidence_range)
        with Progress("Exporting point cloud"):
            xyz, colors, point_frames = self._sample_points(arrays)
            self._write_ply(output / "point_cloud.ply", xyz, colors)
        metadata: dict[str, Any] = {
            "format_version": 1, "units": sequence.units,
            "meters_per_unit": sequence.meters_per_unit,
            "camera_convention": "camera_to_world; OpenCV camera X right, Y down, Z forward",
            "depth_convention": "positive camera Z on processed RGB grid",
            "confidence_convention": "raw VGGT-Omega 1+exp(logit) score; not a probability",
            "intrinsics_grid": "processed RGB; do not apply directly to original frames",
            "confidence_percentile": self.config.confidence_percentile,
            "depth_preview_range": depth_range, "confidence_preview_range": confidence_range,
            "preview_note": "Sequence-wide 2nd–98th percentiles; clipping affects PNGs only",
            "point_cloud_note": "Confidence-filtered, subsampled; overlapping surfaces are not fused",
            "contact_sheet_note": "Overview of up to 24 evenly spaced frames; individual PNGs include every frame",
            "provenance": dict(provenance or {}), "frames": records,
        }
        write_json(output / "metadata.json", metadata)
        with Progress("Writing HTML viewer"):
            self._write_viewer(output, metadata, xyz, colors, point_frames)
        return metadata

    @staticmethod
    def _write_arrays(sequence: GeometrySequence, arrays: GeometryArrays, output: Path) -> None:
        np.savez_compressed(
            output / "geometry.npz", depth=arrays.depth, confidence=arrays.confidence,
            valid_mask=arrays.valid, intrinsics=arrays.intrinsics, camera_to_world=arrays.poses,
            processed_rgb=arrays.rgb, frame_indices=np.array([f.frame_index for f in sequence.frames]),
            units=np.array(sequence.units),
        )

    @staticmethod
    def _write_previews(
        sequence: GeometrySequence, arrays: GeometryArrays, output: Path,
        depth_range: tuple[float, float], confidence_range: tuple[float, float],
    ) -> list[dict[str, Any]]:
        (output / "previews").mkdir()
        # JPEG has a dimension limit; keep the overview bounded for long clips.
        overview_indices = np.linspace(0, len(sequence) - 1, min(24, len(sequence)), dtype=int)
        overview_rows = {int(index): row for row, index in enumerate(overview_indices)}
        sheet = Image.new("RGB", (720, len(overview_rows) * 164), "#151b25")
        draw = ImageDraw.Draw(sheet)
        records: list[dict[str, Any]] = []
        with tracked(sequence.frames, "Saving previews", len(sequence), "frame") as pending:
            for i, frame in enumerate(pending):
                prefix = f"{frame.frame_index:06d}"
                names = {kind: f"previews/{prefix}_{kind}.png" for kind in ("rgb", "depth", "confidence")}
                depth_rgb = _colorize(arrays.depth[i], arrays.valid[i], *depth_range)
                conf_rgb = _colorize(
                    -arrays.confidence[i], np.isfinite(arrays.confidence[i]),
                    -confidence_range[1], -confidence_range[0],
                )
                panels = (("rgb", arrays.rgb[i]), ("depth", depth_rgb), ("confidence", conf_rgb))
                for column, (kind, pixels) in enumerate(panels):
                    image = Image.fromarray(pixels)
                    image.save(output / names[kind])
                    if i in overview_rows:
                        row = overview_rows[i]
                        image.thumbnail((236, 138))
                        sheet.paste(image, (column * 240 + 2, row * 164 + 24))
                        draw.text((column * 240 + 4, row * 164 + 5), f"Frame {frame.frame_index}: {kind}", fill="white")
                mask = torch.from_numpy(arrays.valid[i]).to(frame.depth.device)
                stats = depth_percentiles(frame.depth, mask)
                records.append({
                    "frame_index": frame.frame_index, "source": sequence.source_names[i],
                    "original_size_hw": sequence.original_sizes_hw[i],
                    "processed_size_hw": frame.image_size_hw,
                    "confidence_threshold": arrays.thresholds[i],
                    "valid_fraction": float(arrays.valid[i].mean()),
                    "depth_p10": stats[10], "depth_p50": stats[50], "depth_p90": stats[90],
                    "intrinsics": arrays.intrinsics[i].tolist(), "camera_to_world": arrays.poses[i].tolist(),
                    "previews": names,
                })
        sheet.save(output / "contact_sheet.jpg", quality=90)
        return records

    def _sample_points(self, arrays: GeometryArrays) -> tuple[NDArray[np.float64], ByteArray, NDArray[np.int64]]:
        rng = np.random.default_rng(0)
        per_frame, remainder = divmod(self.config.max_points, len(arrays.depth))
        points: list[NDArray[np.float64]] = []
        colors: list[ByteArray] = []
        frame_ids: list[NDArray[np.int64]] = []
        for i in range(len(arrays.depth)):
            indices = np.flatnonzero(arrays.valid[i])
            budget = per_frame + (i < remainder)
            if len(indices) > budget:
                indices = rng.choice(indices, budget, replace=False)
            yy, xx = np.unravel_index(indices, arrays.valid[i].shape)
            pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
            camera_points = (pixels @ np.linalg.inv(arrays.intrinsics[i].astype(np.float64)).T)
            camera_points *= arrays.depth[i, yy, xx, None]
            world_points = camera_points @ arrays.poses[i, :3, :3].T + arrays.poses[i, :3, 3]
            finite = np.isfinite(world_points).all(axis=1)
            points.append(world_points[finite])
            colors.append(arrays.rgb[i, yy, xx][finite])
            frame_ids.append(np.full(int(finite.sum()), i, dtype=np.int64))
        return np.concatenate(points), np.concatenate(colors), np.concatenate(frame_ids)

    @staticmethod
    def _write_ply(path: Path, xyz: NDArray[np.float64], colors: ByteArray) -> None:
        with path.open("w", encoding="utf-8") as file:
            file.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\n")
            file.write("property float x\nproperty float y\nproperty float z\n")
            file.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for point, color in zip(xyz, colors, strict=True):
                file.write(f"{point[0]:.6g} {point[1]:.6g} {point[2]:.6g} "
                           f"{color[0]} {color[1]} {color[2]}\n")

    @staticmethod
    def _write_viewer(
        output: Path, metadata: Mapping[str, Any], xyz: NDArray[np.float64], colors: ByteArray,
        point_frames: NDArray[np.int64],
    ) -> None:
        payload = dict(metadata)
        payload["point_frames"] = point_frames.tolist()
        payload["points"] = np.column_stack([xyz, colors]).round(5).tolist()
        embedded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
        template = Path(__file__).with_name("templates") / "geometry.html"
        html = template.read_text(encoding="utf-8").replace("__GEOMETRY_DATA__", embedded)
        (output / "index.html").write_text(html, encoding="utf-8")


def save_geometry_report(
    sequence: GeometrySequence, output: str | Path, *, confidence_percentile: float = 20,
    max_points: int = 24000, provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility entry point for callers of the initial demo implementation."""
    writer = GeometryReportWriter(PreviewConfig(confidence_percentile, max_points))
    return writer.write(sequence, Path(output), provenance=provenance)
