"""Memory-aware GPU workers, overlapping OOM retries and ordered reconstruction."""

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import gc
import json
import logging
import math
from pathlib import Path
from threading import Event, Lock
from uuid import uuid4

import torch
from stereoforge.utils.progress import Progress, tracked

from .alignment import OverlapAlignmentError, align_overlap
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
        return [DeviceBudget("cpu", "CPU", 0, 0, config.chunk_max_frames or 32)]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check Docker GPU access")
    indices = (range(torch.cuda.device_count()) if config.device == "cuda"
               else [int(config.device.split(":")[1])])
    budgets = []
    for index in indices:
        free, total = torch.cuda.mem_get_info(index)
        # Bootstrap only: measured peaks drive subsequent section sizes.
        usable = max(0, total * config.gpu_memory_fraction - (total - free) - 6 * GIB)
        estimated = int(usable / (0.25 * GIB) * (512 / config.image_resolution) ** 2)
        count = max(config.chunk_overlap + 2, min(config.chunk_max_frames or max(1, estimated), estimated))
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
        cancelled = Event()
        lock = Lock()
        cursor = 0
        covered = bytearray(len(paths))
        measurements: list[dict] = []

        def claim(capacity: int) -> Section | None:
            nonlocal cursor
            with lock:
                if cursor >= len(paths) or cancelled.is_set():
                    return None
                section = Section(cursor, min(len(paths), cursor + capacity))
                cursor = (section.stop if section.stop == len(paths)
                          else section.stop - self.config.chunk_overlap)
                return section

        def mark_completed(section: Section) -> None:
            # Only successfully saved predictions count. Shared frames and OOM
            # retries must not inflate the fixed full-video denominator.
            with lock:
                newly_completed = section.stop - section.start - sum(covered[section.start:section.stop])
                covered[section.start:section.stop] = b"\x01" * (section.stop - section.start)
                overall.advance(newly_completed)

        def run_worker(budget: DeviceBudget) -> tuple[list[Section], list[dict]]:
            try:
                return self._worker(budget, paths, directory, cancelled, claim, measurements,
                                    len(budgets), mark_completed)
            except BaseException:
                cancelled.set()
                raise

        with Progress("VGGT inference", len(paths), "frame") as overall, ThreadPoolExecutor(
            max_workers=len(budgets), thread_name_prefix="geometry-gpu",
        ) as pool:
            overall.status("unique frames saved across all GPUs")
            futures = [pool.submit(run_worker, budget) for budget in budgets]
            try:
                results = [future.result() for future in futures]
            except BaseException:
                cancelled.set()
                raise
        completed = sorted(item for sections, _ in results for item in sections)
        retries = [item for _, attempts in results for item in attempts]
        LOGGER.info("VGGT models unloaded and CUDA caches released; merging reconstruction on CPU")
        try:
            sequence, merges = self._merge(completed, directory, len(paths))
        except (ValueError, RuntimeError) as exc:
            # Move outside staged_output before its cleanup removes the failed run.
            retained = directory.parent.parent / "failed_reconstructions" / uuid4().hex
            retained.parent.mkdir(parents=True, exist_ok=True)
            directory.rename(retained)
            report = {"error": str(exc), "frame_count": len(paths),
                      "sections": [asdict(section) for section in completed],
                      "devices": [asdict(budget) for budget in budgets],
                      "overlap": self.config.chunk_overlap,
                      "alignment": exc.diagnostics if isinstance(exc, OverlapAlignmentError) else None}
            try:
                (retained / "failure.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            except OSError:
                LOGGER.exception("Could not write merge diagnostics; section tensors retained in %s", retained)
            raise ValueError(f"{exc}\nSection tensors and merge diagnostics retained in {retained}") from exc
        return sequence, {"devices": [asdict(d) for d in budgets], "oom_retries": retries,
                          "memory_measurements": sorted(measurements, key=lambda m: (m["start"], m["stop"])),
                          "gpu_memory_fraction": self.config.gpu_memory_fraction,
                          "sections": merges, "overlap": self.config.chunk_overlap,
                          "alignment": "sequential robust Sim(3); first section anchors scale and world",
                          "overlap_policy": "retain earlier section; append each new frame once",
                          "limitations": "VRAM target is not GPU compute utilization; external allocations can change. No loop closure/global optimization."}

    def _worker(self, budget: DeviceBudget, paths: tuple[Path, ...], directory: Path,
                cancelled: Event, claim: Callable[[int], Section | None],
                measurements: list[dict], device_count: int,
                mark_completed: Callable[[Section], None],
                ) -> tuple[list[Section], list[dict]]:
        completed, retries = [], []
        cuda = budget.device != "cpu"
        context = torch.cuda.device(budget.device) if cuda else nullcontext()
        minimum = self.config.chunk_overlap + 1
        ceiling = self.config.chunk_max_frames or len(paths)
        # Share initial work across available GPUs; subsequent claims use actual peaks.
        capacity = max(minimum, min(budget.initial_frames,
                                   math.ceil(len(paths) / device_count) + self.config.chunk_overlap, ceiling))
        failed_size = ceiling + 1
        with context:
            try:
                with Progress(budget.device, unit="section") as progress, VGGTOmegaGeometryEstimator(
                    self.checkpoint, device=budget.device, image_resolution=self.config.image_resolution,
                    preprocess_mode=self.config.preprocess_mode,
                ) as estimator:
                    LOGGER.debug("%s: %s, initial section size %d", budget.device, budget.name, capacity)

                    def process(section: Section) -> None:
                        nonlocal capacity, failed_size
                        if cancelled.is_set():
                            return
                        size = section.stop - section.start
                        if size > capacity:
                            split(section)
                            return
                        allowed = 0
                        if cuda:
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            free, total = torch.cuda.mem_get_info()
                            external = max(0, total - free - torch.cuda.memory_reserved())
                            allowed = int(total * self.config.gpu_memory_fraction) - external
                            if allowed <= 0:
                                raise RuntimeError(f"{budget.device}: other allocations already consume the VRAM budget")
                            # This bounds PyTorch's allocator, not allocations by other processes/libraries.
                            torch.cuda.set_per_process_memory_fraction(allowed / total)
                            torch.cuda.reset_peak_memory_stats()
                        failed = False
                        try:
                            sequence = estimator.predict(
                                paths[section.start:section.stop], section.start,
                                status=lambda detail: progress.status(f"frames {section.start}:{section.stop} | {detail}"),
                            )
                        except GeometryOutOfMemoryError:
                            failed = True
                        if failed:
                            gc.collect()
                            if cuda:
                                torch.cuda.empty_cache()
                            retries.append({"device": budget.device, **asdict(section)})
                            if size <= minimum:
                                raise RuntimeError(f"{budget.device}: cannot fit {size} frames within the VRAM budget; "
                                                   "free memory or reduce chunk_overlap")
                            failed_size = min(failed_size, size)
                            capacity = (size + self.config.chunk_overlap + 1) // 2
                            LOGGER.warning("%s: allocation limit reached; retrying smaller sections", budget.device)
                            split(section)
                            return
                        if cuda:
                            torch.cuda.synchronize()
                            peak = torch.cuda.max_memory_allocated()
                            reserved = torch.cuda.max_memory_reserved()
                            # Conservative square-root growth accounts for nonlinear attention cost.
                            factor = min(1.5, math.sqrt(max(1, allowed) / max(1, peak)))
                            proposal = int(size * factor)
                            capacity = max(minimum, min(ceiling, failed_size - 1, proposal))
                            measurements.append({"device": budget.device, **asdict(section),
                                                 "allocator_budget_bytes": allowed, "peak_allocated_bytes": peak,
                                                 "peak_reserved_bytes": reserved, "next_capacity": capacity})
                        progress.status(f"frames {section.start}:{section.stop} | saving predictions")
                        _save(sequence, directory / f"{section.start}_{section.stop}.pt")
                        completed.append(section)
                        mark_completed(section)
                        progress.advance()

                    def split(section: Section) -> None:
                        middle = (section.start + section.stop - self.config.chunk_overlap) // 2
                        process(Section(section.start, middle + self.config.chunk_overlap))
                        process(Section(middle, section.stop))

                    while (section := claim(capacity)) is not None:
                        process(section)
            finally:
                # The estimator context has dropped model references before this point.
                gc.collect()
                if cuda:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.set_per_process_memory_fraction(1.0)
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
                    diagnostics = {"failed_section": asdict(section), "merged_sections": records,
                                   "overlap": exc.diagnostics if isinstance(exc, OverlapAlignmentError) else None}
                    raise OverlapAlignmentError(
                        f"Cannot merge section [{section.start}, {section.stop}): {exc}", diagnostics) from exc
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
        if len(frames) != count:
            raise ValueError(f"Incomplete reconstruction: {len(frames)}/{count} frames")
        scale = self.config.meters_per_unit
        if scale is not None:
            # Calibration belongs to the first section's scale, after alignment.
            for index, frame in frames.items():
                pose = frame.camera_to_world.clone()
                pose[:3, 3] *= scale
                frames[index] = replace(frame, depth=frame.depth * scale, camera_to_world=pose)
        sequence = GeometrySequence(tuple(frames.values()), torch.stack(rgb), tuple(names), tuple(sizes),
                                    "meters" if scale is not None else "reconstruction_units", scale)
        # Keep every input until reconstruction and validation have both succeeded.
        for section in sections:
            (directory / f"{section.start}_{section.stop}.pt").unlink()
        directory.rmdir()
        return sequence, records
