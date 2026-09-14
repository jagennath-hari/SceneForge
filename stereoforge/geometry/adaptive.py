"""Memory-aware GPU workers, overlapping OOM retries and ordered reconstruction."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import gc
import logging
from pathlib import Path
from threading import Event

import torch
from stereoforge.utils.progress import Progress, tracked

from .alignment import align_overlap
from .config import GeometryConfig
from .types import FrameGeometry, GeometrySequence
from .vggt_omega import GeometryOutOfMemoryError, VGGTOmegaGeometryEstimator

LOGGER = logging.getLogger(__name__)
GIB = 1024 ** 3


@dataclass(frozen=True, slots=True)
class DeviceBudget:
    device: str
    name: str
    free_bytes: int
    total_bytes: int
    initial_frames: int


@dataclass(frozen=True, slots=True, order=True)
class Section:
    start: int
    stop: int


def discover_devices(config: GeometryConfig) -> list[DeviceBudget]:
    if config.device == "cpu":
        return [DeviceBudget("cpu", "CPU", 0, 0, config.chunk_max_frames)]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check Docker GPU access")
    indices = (range(torch.cuda.device_count()) if config.device == "cuda"
               else [int(config.device.split(":")[1])])
    budgets = []
    for index in indices:
        free, total = torch.cuda.mem_get_info(index)
        # Starting heuristic, not a linear memory prediction. The earlier 0.75
        # GiB/frame allowance selected only ~42 frames on a 48 GiB A6000.
        # Try larger sections; OOM splitting remains the actual capacity check.
        usable = max(0, free * config.gpu_memory_fraction - 6 * GIB)
        estimated = int(usable / (0.25 * GIB) * (512 / config.image_resolution) ** 2)
        count = max(config.chunk_overlap + 2, min(config.chunk_max_frames, estimated))
        budgets.append(DeviceBudget(f"cuda:{index}", torch.cuda.get_device_name(index), free, total, count))
    return budgets


def _save(sequence: GeometrySequence, path: Path) -> None:
    # Tensor-only dictionaries can be loaded with weights_only=True.
    torch.save({"frames": [dict(frame_index=f.frame_index, depth=f.depth.clone(),
                               confidence=f.confidence.clone(), intrinsics=f.intrinsics.clone(),
                               camera_to_world=f.camera_to_world.clone()) for f in sequence.frames],
                "rgb": sequence.processed_rgb, "names": sequence.source_names,
                "sizes": sequence.original_sizes_hw}, path)


def _load(path: Path) -> GeometrySequence:
    data = torch.load(path, map_location="cpu", weights_only=True)
    return GeometrySequence(tuple(FrameGeometry(**f) for f in data["frames"]),
                            data["rgb"], data["names"], data["sizes"])


class AdaptiveGeometryEstimator:
    """One model per visible GPU; disk-spooled sections bound pending CPU results.

    Final merging/report export still use host RAM proportional to the video.
    All transforms anchor to the first section. This is sequential registration,
    not global bundle adjustment or loop closure.
    """

    def __init__(self, checkpoint: Path, config: GeometryConfig) -> None:
        self.checkpoint, self.config = checkpoint, config

    def predict(self, paths: tuple[Path, ...], directory: Path) -> tuple[GeometrySequence, dict]:
        if not paths:
            raise ValueError("No frames supplied")
        budgets = discover_devices(self.config)
        LOGGER.info("Using %d device(s); total VRAM %.1f GiB, currently free %.1f GiB (not pooled)",
                    len(budgets), sum(d.total_bytes for d in budgets) / GIB,
                    sum(d.free_bytes for d in budgets) / GIB)
        directory.mkdir()
        assignments: list[list[Section]] = [[] for _ in budgets]
        start, worker = 0, 0
        while start < len(paths):
            stop = min(len(paths), start + budgets[worker].initial_frames)
            assignments[worker].append(Section(start, stop))
            if stop == len(paths):
                break
            start = stop - self.config.chunk_overlap
            worker = (worker + 1) % len(budgets)
        cancelled = Event()

        def run_worker(budget: DeviceBudget, sections: list[Section]) -> tuple[list[Section], list[dict]]:
            try:
                return self._worker(budget, sections, paths, directory, cancelled)
            except BaseException:
                cancelled.set()
                raise

        with ThreadPoolExecutor(max_workers=len(budgets), thread_name_prefix="geometry-gpu") as pool:
            futures = [pool.submit(run_worker, budget, sections)
                       for budget, sections in zip(budgets, assignments, strict=True) if sections]
            try:
                results = [future.result() for future in futures]
            except BaseException:
                cancelled.set()
                raise
        completed = sorted(item for sections, _ in results for item in sections)
        retries = [item for _, attempts in results for item in attempts]
        sequence, merges = self._merge(completed, directory, len(paths))
        return sequence, {"devices": [asdict(d) for d in budgets], "oom_retries": retries,
                          "assignments": [{"device": budget.device, "sections": [asdict(s) for s in assigned]}
                                          for budget, assigned in zip(budgets, assignments, strict=True)],
                          "sections": merges, "overlap": self.config.chunk_overlap,
                          "alignment": "sequential robust Sim(3); first section anchors scale and world",
                          "overlap_policy": "retain earlier section; append each new frame once",
                          "limitations": "No loop closure/global optimization; accumulated drift and seams remain possible"}

    def _worker(self, budget: DeviceBudget, sections: list[Section], paths: tuple[Path, ...],
                directory: Path, cancelled: Event) -> tuple[list[Section], list[dict]]:
        completed, retries = [], []
        context = torch.cuda.device(budget.device) if budget.device != "cpu" else nullcontext()
        with context, Progress(budget.device, len(sections), "section") as progress, VGGTOmegaGeometryEstimator(
            self.checkpoint, device=budget.device, image_resolution=self.config.image_resolution,
            preprocess_mode=self.config.preprocess_mode,
        ) as estimator:
            LOGGER.info("%s: %s, free %.1f/%.1f GiB, starting section size %d",
                        budget.device, budget.name, budget.free_bytes / GIB,
                        budget.total_bytes / GIB, budget.initial_frames)
            capacity = budget.initial_frames

            def process(section: Section) -> None:
                nonlocal capacity
                if cancelled.is_set():
                    return
                size = section.stop - section.start
                if size > capacity:
                    split(section)
                    return
                failed = False
                try:
                    progress.status(f"frames {section.start}:{section.stop} | loading/preprocessing/inference")
                    LOGGER.debug("%s: inferring frames [%d, %d)", budget.device, section.start, section.stop)
                    sequence = estimator.predict(
                        paths[section.start:section.stop], section.start,
                        status=lambda detail: progress.status(f"frames {section.start}:{section.stop} | {detail}"),
                    )
                except GeometryOutOfMemoryError:
                    # Leave the except block before clearing the allocator: its
                    # traceback otherwise retains failed inference tensors.
                    failed = True
                if failed:
                    gc.collect()
                    if budget.device != "cpu":
                        torch.cuda.empty_cache()
                    retries.append({"device": budget.device, **asdict(section)})
                    if size <= self.config.chunk_overlap + 1:
                        raise RuntimeError(
                            f"{budget.device}: cannot fit minimum overlapping section ({size} frames). "
                            "Free GPU memory or reduce geometry.chunk_overlap; no frames were dropped."
                        )
                    capacity = (size + self.config.chunk_overlap + 1) // 2
                    LOGGER.warning("%s: OOM; reducing section capacity to %d", budget.device, capacity)
                    progress.status(f"OOM retry | capacity {capacity} frames")
                    split(section)
                    return
                progress.status(f"frames {section.start}:{section.stop} | saving predictions")
                _save(sequence, directory / f"{section.start}_{section.stop}.pt")
                completed.append(section)
                progress.advance()

            def split(section: Section) -> None:
                progress.add_work()  # One pending section becomes two.
                # Children overlap by exactly the configured amount and both
                # shrink. Recursion terminates at overlap + 1 frames.
                middle = (section.start + section.stop - self.config.chunk_overlap) // 2
                process(Section(section.start, middle + self.config.chunk_overlap))
                process(Section(middle, section.stop))

            for section in sections:
                process(section)
        return completed, retries

    def _merge(self, sections: list[Section], directory: Path, count: int) -> tuple[GeometrySequence, list[dict]]:
        frames, rgb, names, sizes, records = {}, [], [], [], []
        with tracked(sections, "Merging reconstruction", len(sections), "section") as pending:
            for section in pending:
                path = directory / f"{section.start}_{section.stop}.pt"
                local = _load(path)
                try:
                    transform = align_overlap(frames, local) if frames else None
                except (ValueError, RuntimeError) as exc:
                    raise ValueError(f"Cannot merge section [{section.start}, {section.stop}): {exc}") from exc
                records.append({**asdict(section), "transform": transform.as_dict() if transform else None})
                LOGGER.debug("Merging frames [%d, %d)%s", section.start, section.stop,
                            f", overlap error={transform.relative_error:.4f}" if transform else " (world anchor)")
                for i, frame in enumerate(local.frames):
                    if frame.frame_index in frames:
                        continue
                    if frame.frame_index != len(frames):
                        raise ValueError("Section merge would leave a missing frame")
                    frames[frame.frame_index] = transform.apply(frame) if transform else frame
                    rgb.append(local.processed_rgb[i].clone())
                    names.append(local.source_names[i])
                    sizes.append(local.original_sizes_hw[i])
                path.unlink()
        if len(frames) != count:
            raise ValueError(f"Incomplete reconstruction: {len(frames)}/{count} frames")
        scale = self.config.meters_per_unit
        if scale is not None:
            # Calibration belongs to the first section's scale, after alignment.
            for index, frame in frames.items():
                pose = frame.camera_to_world.clone()
                pose[:3, 3] *= scale
                frames[index] = replace(frame, depth=frame.depth * scale, camera_to_world=pose)
        directory.rmdir()
        return GeometrySequence(tuple(frames.values()), torch.stack(rgb), tuple(names), tuple(sizes),
                                "meters" if scale is not None else "reconstruction_units", scale), records
