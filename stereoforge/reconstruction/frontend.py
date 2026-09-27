"""Shared video reconstruction frontend."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
from PIL import Image
import torch

from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import Progress, tracked
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

    def _prepare_images(self, paths: list[Path], reused: dict[Path, Path] | None = None,
                        description: str = "Preparing feature images") -> dict[int, Path]:
        from vggt_omega.utils.load_fn import load_and_preprocess_images

        folder = self.output / "processed"
        folder.mkdir(exist_ok=True)
        result = {}
        sizes = set()
        if self.visualization is not None:
            self.visualization.activity('Preparing feature images — fixed overview', overview=True)
        with tracked(enumerate(paths), description, len(paths), "frame") as pending:
            for index, path in pending:
                destination = folder / f"{index:06d}.png"
                if self.visualization is not None:
                    self.visualization.activity(f"Preparing feature image: {index+1}/{len(paths)}", [index])
                if not destination.exists() and reused is not None and path in reused:
                    VideoFrameSampler._link_or_copy(reused[path], destination)
                if not destination.exists():
                    tensor = load_and_preprocess_images([str(path)], mode="balanced", image_resolution=512)[0]
                    pixels = (tensor.numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
                    temporary = destination.with_suffix(".partial")
                    Image.fromarray(pixels).save(temporary, format="PNG", compress_level=1)
                    temporary.rename(destination)
                with Image.open(destination) as image:
                    sizes.add(image.size)
                result[index] = destination
                if self.visualization is not None:
                    self.visualization.keyframe(index, destination)
                    self.visualization.activity(f"Feature image ready: {index+1}/{len(paths)}", [index], ready=True)
        if self.visualization is not None:
            self.visualization.event(f"Feature images ready: {len(paths)}/{len(paths)}")
        if len(sizes) != 1:
            raise ValueError("Video input must use a uniform processed image size")
        self.size_wh = next(iter(sizes))
        return result

    def _match(self, count: int, reused: dict[tuple[int, int], dict] | None = None,
               description: str = "Verifying temporal image pairs") -> list[Path]:
        executable = shutil.which("stereoforge-match-pairs")
        if executable is None:
            raise RuntimeError("Rebuild Docker to install the pair matcher with global feature IDs")
        folder = self.output / "matching"
        folder.mkdir(exist_ok=True)
        models = None
        reused = reused or {}
        prior_targets: dict[int, list[int]] = {}
        for a, b in reused:
            prior_targets.setdefault(a, []).append(b)
        result = []
        width, height = self.size_wh
        batches = range(0, max(count - 1, 0), 64)
        plans = []
        for start in batches:
            pairs = sorted({(a, b) for a in range(start, min(start+64, count))
                            for b in range(a+1, min(a+self.options.neighbors+1, count))}
                           | {(a, b) for a in range(start, min(start+64, count))
                              for b in prior_targets.get(a, ())})
            destination = folder / f"{start:06d}.jsonl"
            missing = [] if destination.is_file() else [pair for pair in pairs if pair not in reused]
            plans.append((start, destination, pairs, missing))
        # Engine preparation has its own bar. Finish it before opening the
        # matching bar, including when only a middle batch needs new inference.
        if any(missing for _, _, _, missing in plans):
            if self.visualization is not None:
                self.visualization.event("Preparing RaCo–ALIKED + LightGlue+ for new image pairs")
            models = VideoFrameSampler.model_arguments(self.keyframe_config)[1]
        total_new = sum(len(missing) for _, _, _, missing in plans)
        total_pairs = sum(len(pairs) for _, _, pairs, _ in plans)
        completed_new = 0
        with Progress(description, total=total_pairs, unit="pair") as progress:
            for batch_index, (start, destination, pairs, missing) in enumerate(plans):
                progress.status(f"batch {batch_index+1}/{len(plans)} · new {completed_new}/{total_new} · "
                                f"{total_pairs-total_new} reusable")
                if destination.is_file():
                    result.append(destination)
                    if self.visualization is not None:
                        self.visualization.matched_batch(destination)
                    progress.advance(len(pairs))
                    continue
                if not pairs:
                    continue
                request = folder / f"{start:06d}.request.json"
                write_json(request, {"pairs": [{"source": a, "target": b, "width": width, "height": height,
                             "source_path": str(self.images[a]), "target_path": str(self.images[b])} for a, b in missing]})
                temporary = destination.with_suffix(".partial_" + uuid4().hex[:8])
                command = [executable, str(request), str(self.keyframe_config), models or "", str(temporary)]
                if self.device:
                    command.append(str(self.device))
                if self.visualization is not None:
                    sample = pairs[:16]
                    self.visualization.activity(
                        f"Matching batch {start//64+1}/{len(batches)} — requested pairs, awaiting verification",
                        sorted({f for pair in sample for f in pair}), sample)
                if missing:
                    with destination.with_suffix(".log").open("w") as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                # Verify identity and schema before marking this batch reusable.
                records = [json.loads(line) for line in temporary.read_text().splitlines()] if missing else []
                if [(r["source"], r["target"]) for r in records] != missing:
                    raise ValueError("Pair matcher returned incomplete or unexpected frame IDs")
                if any("source_feature" not in m for r in records for m in r["matches"]):
                    raise ValueError("Rebuild Docker: pair matcher lacks stable global feature IDs")
                combined = {pair: reused[pair] for pair in pairs if pair in reused}
                combined.update({(record["source"], record["target"]): record for record in records})
                with temporary.open("w") as stream:
                    for pair in pairs:
                        stream.write(json.dumps(combined[pair]) + "\n")
                temporary.rename(destination)
                result.append(destination)
                if self.visualization is not None:
                    self.visualization.matched_batch(destination)
                completed_new += len(missing)
                progress.advance(len(pairs))
        return result
