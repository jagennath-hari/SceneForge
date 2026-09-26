"""Independent small VGGT clusters scheduled across visible GPUs."""

from dataclasses import replace
from collections.abc import Callable
import logging
import multiprocessing as mp
from multiprocessing.queues import Queue
from pathlib import Path
from queue import Empty

import torch

from stereoforge.geometry.storage import save_sequence
from stereoforge.geometry.vggt_omega import VGGTOmegaGeometryEstimator
from stereoforge.utils.progress import Progress


def _worker(device: str, checkpoint: str, paths: list[str], jobs: list[tuple[int, list[int]]],
            directory: str, messages: Queue) -> None:
    logging.getLogger("dinov3").setLevel(logging.WARNING)
    try:
        if device.startswith("cuda"):
            torch.cuda.set_device(device)
            torch.cuda.set_per_process_memory_fraction(0.9, device)
        with VGGTOmegaGeometryEstimator(checkpoint, device=device) as estimator:
            for identifier, ids in jobs:
                result = estimator.predict([paths[i] for i in ids])
                result = replace(result, frames=tuple(replace(f, frame_index=ids[i])
                                                     for i, f in enumerate(result.frames)))
                destination = Path(directory) / f"{identifier}.pt"
                temporary = destination.with_suffix(".partial")
                save_sequence(result, temporary)
                temporary.rename(destination)
                del result
                messages.put(("done", identifier))
    except BaseException as exc:
        messages.put(("error", f"{device}: {type(exc).__name__}: {exc}"))
        raise


def infer_clusters(checkpoint: Path, paths: list[Path], leaves: list,
                   output: Path, devices: list[str], on_complete: Callable[[int], None] | None = None) -> None:
    output.mkdir(exist_ok=True)
    jobs = [(node.identifier, node.frames) for node in leaves if not (output / f"{node.identifier}.pt").is_file()]
    if on_complete is not None:
        pending_ids = {identifier for identifier, _ in jobs}
        for node in leaves:
            if node.identifier not in pending_ids:
                on_complete(node.identifier)
    if not jobs:
        return
    context = mp.get_context("spawn")
    messages = context.Queue()
    workers = []
    try:
        for index, device in enumerate(devices[:len(jobs)]):
            process = context.Process(target=_worker, args=(device, str(checkpoint), list(map(str, paths)),
                                      jobs[index::len(devices[:len(jobs)])], str(output), messages))
            process.start()
            workers.append(process)
        with Progress("VGGT cluster inference", total=len(jobs), unit="cluster") as progress:
            completed = 0
            while completed < len(jobs):
                try:
                    kind, value = messages.get(timeout=1)
                except Empty:
                    if any(p.exitcode not in (None, 0) for p in workers) or all(not p.is_alive() for p in workers):
                        raise RuntimeError("VGGT worker exited before publishing all clusters; saved clusters are retained")
                    continue
                if kind == "error":
                    raise RuntimeError(str(value))
                if on_complete is not None:
                    on_complete(value)
                completed += 1
                progress.advance(1)
    finally:
        for process in workers:
            if process.is_alive():
                process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
        messages.close()
