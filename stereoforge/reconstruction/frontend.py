"""Shared video reconstruction frontend and sparse viewer export."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
from PIL import Image
import torch

from stereoforge.refinement.sparse_model import SparseModel
from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import Progress, tracked
from stereoforge.utils.visualization import GeometryReportWriter
from stereoforge.video.sampling import VideoFrameSampler

if TYPE_CHECKING:
    from .windowed import WindowOptions


class ReconstructionFrontend:
    def __init__(self, options: WindowOptions, output: Path, keyframe_config: Path) -> None:
        self.options, self.output, self.keyframe_config = options, output, keyframe_config
        self.visualization = None
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for reconstruction matching and optimization")
        self.devices = ([f"cuda:{i}" for i in range(torch.cuda.device_count())] if options.device == "cuda"
                        else [options.device])
        self.device = int(self.devices[0].split(":")[1])
        if self.device >= torch.cuda.device_count():
            raise ValueError("Requested device is outside the visible CUDA devices")
        executable = shutil.which("stereoforge-match-pairs")
        if executable is None:
            raise RuntimeError("Rebuild Docker to install the reconstruction pair matcher")
        capabilities = subprocess.run([executable, "--capabilities"], capture_output=True, text=True)
        if capabilities.returncode or not json.loads(capabilities.stdout).get("stable_feature_ids"):
            raise RuntimeError("Pair matcher needs rebuilding: missing stable feature ID support")

    def _prepare_images(self, paths: list[Path]) -> dict[int, Path]:
        from vggt_omega.utils.load_fn import load_and_preprocess_images

        folder = self.output / "processed"
        folder.mkdir(exist_ok=True)
        result = {}
        sizes = set()
        with tracked(enumerate(paths), "Preparing feature images", len(paths), "frame") as pending:
            for index, path in pending:
                destination = folder / f"{index:06d}.png"
                if not destination.exists():
                    tensor = load_and_preprocess_images([str(path)], mode="balanced", image_resolution=512)[0]
                    pixels = (tensor.numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
                    temporary = destination.with_suffix(".partial")
                    Image.fromarray(pixels).save(temporary, format="PNG", compress_level=1)
                    temporary.rename(destination)
                with Image.open(destination) as image:
                    sizes.add(image.size)
                result[index] = destination
                if self.visualization is not None and (index % 20 == 0 or index == len(paths)-1):
                    self.visualization.event(f"Preparing feature images: {index+1}/{len(paths)}; processed keyframe {index}", destination)
        if len(sizes) != 1:
            raise ValueError("Video input must use a uniform processed image size")
        self.size_wh = next(iter(sizes))
        return result

    def _match(self, count: int) -> list[Path]:
        executable = shutil.which("stereoforge-match-pairs")
        if executable is None:
            raise RuntimeError("Rebuild Docker to install the pair matcher with global feature IDs")
        folder = self.output / "matching"
        folder.mkdir(exist_ok=True)
        models = None
        result = []
        width, height = self.size_wh
        batches = range(0, max(count - 1, 0), 64)
        if any(not (folder / f"{start:06d}.jsonl").is_file() for start in batches):
            models = VideoFrameSampler.model_arguments(self.keyframe_config)[1]
        with Progress("Verifying temporal image pairs", total=len(batches), unit="batch") as progress:
            for start in batches:
                destination = folder / f"{start:06d}.jsonl"
                if destination.is_file():
                    result.append(destination)
                    if self.visualization is not None:
                        self.visualization.match_batch(destination, self.images)
                    progress.status(f"reused batch · source frames {start}–{min(start+64, count-1)-1}")
                    progress.advance()
                    continue
                pairs = [(a, b) for a in range(start, min(start+64, count))
                         for b in range(a+1, min(a+self.options.neighbors+1, count))]
                if not pairs:
                    continue
                request = folder / f"{start:06d}.request.json"
                write_json(request, {"pairs": [{"source": a, "target": b, "width": width, "height": height,
                             "source_path": str(self.images[a]), "target_path": str(self.images[b])} for a, b in pairs]})
                temporary = destination.with_suffix(".partial_" + uuid4().hex[:8])
                command = [executable, str(request), str(self.keyframe_config), models, str(temporary)]
                if self.device:
                    command.append(str(self.device))
                progress.status(f"source frames {start}–{min(start+64, count-1)-1}")
                with destination.with_suffix(".log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                # Verify identity and schema before marking this batch reusable.
                records = [json.loads(line) for line in temporary.read_text().splitlines()]
                if [(r["source"], r["target"]) for r in records] != pairs:
                    raise ValueError("Pair matcher returned incomplete or unexpected frame IDs")
                if any("source_feature" not in m for r in records for m in r["matches"]):
                    raise ValueError("Rebuild Docker: pair matcher lacks stable global feature IDs")
                temporary.rename(destination)
                result.append(destination)
                if self.visualization is not None:
                    self.visualization.match_batch(destination, self.images)
                progress.advance()
        return result

    def _viewer(self, model: SparseModel, summary: dict, timestamps: tuple | None,
                directory: Path | None = None) -> None:
        destination = directory if directory is not None else self.output
        frames, order = [], {f: i for i, f in enumerate(sorted(model.cameras))}
        for frame, camera in sorted(model.cameras.items()):
            frames.append({"frame_index": frame, "source": f"Keyframe {frame}",
                           "processed_size_hw": camera.size_hw, "camera_to_world": camera.pose.tolist(),
                           "intrinsics": camera.intrinsics.tolist(), "valid_fraction": 0, "depth_p50": None,
                           "previews": {"rgb": os.path.relpath(self.images[frame], destination)}})
        metadata = {"format_version": 1, "units": "reconstruction_units", "meters_per_unit": None,
                    "reconstruction_name": summary.get("reconstruction_name", "StereoForge reconstruction"), "sparse_only": True,
                    "frames": frames, "input_frames": summary["input_frames"], "sparse_statistics": summary,
                    "dense_depth_refined": False, "provenance": {"video_timestamps_seconds":
                        [timestamps[f] for f in sorted(model.cameras)] if timestamps else None}}
        write_json(destination / "metadata.json", metadata)
        selected = np.linspace(0, len(model.points)-1, min(240000, len(model.points)), dtype=int)
        points = [model.points[i] for i in selected]
        xyz, rgb = np.array([p.xyz for p in points]), np.array([p.rgb for p in points], dtype=np.uint8)
        appeared = np.array([min(order[f] for f in p.observations) for p in points], dtype=np.int64)
        GeometryReportWriter._write_ply(destination / "point_cloud.ply", xyz, rgb)
        GeometryReportWriter._write_viewer(destination, metadata, xyz, rgb, appeared)
