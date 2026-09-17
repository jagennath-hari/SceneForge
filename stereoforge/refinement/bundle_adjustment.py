"""Sparse pyCuSFM bundle adjustment and reconstruction statistics."""

import importlib.util
import os
from pathlib import Path
import signal
import subprocess

import numpy as np

from stereoforge.geometry.config import RefinementConfig
from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import Progress
from .colmap_binary import ColmapBinaryWriter, bundle_adjustment_command
from .colmap_validation import ColmapOutputValidator
from .sparse_model import SparseModel, reprojection_error


def model_statistics(model: SparseModel) -> dict:
    per_frame: dict[int, list[float]] = {f: [] for f in model.cameras}
    for point in model.points:
        for frame, uv in point.observations.items():
            per_frame[frame].append(reprojection_error(model.cameras[frame], point.xyz, uv))
    errors = [error for values in per_frame.values() for error in values]
    finite = np.asarray(errors)[np.isfinite(errors)]
    frames = []
    for frame, values in sorted(per_frame.items()):
        valid = np.asarray(values)[np.isfinite(values)]
        frames.append({"frame": frame, "observations": len(values),
                       "invalid_projections": len(values) - len(valid),
                       "median_reprojection_pixels": float(np.median(valid)) if len(valid) else None})
    return {"registered_frames": len(model.cameras), "landmarks": len(model.points),
            "observations": len(errors), "invalid_projections": len(errors) - len(finite),
            "median_reprojection_pixels": float(np.median(finite)) if len(finite) else None,
            "mean_reprojection_pixels": float(np.mean(finite)) if len(finite) else None,
            "frames": frames}


def require_connected(model: SparseModel) -> None:
    neighbors = {f: set() for f in model.cameras}
    for point in model.points:
        frames = list(point.observations)
        for frame in frames[1:]:
            neighbors[frames[0]].add(frame)
            neighbors[frame].add(frames[0])
    if not neighbors:
        raise ValueError("Empty sparse reconstruction")
    pending, visited = [next(iter(neighbors))], set()
    while pending:
        frame = pending.pop()
        if frame not in visited:
            visited.add(frame)
            pending.extend(neighbors[frame] - visited)
    if len(visited) != len(neighbors):
        raise ValueError("Sparse camera/track graph is disconnected; one Sim(3) cannot fix independent gauges")


class SparseBundleAdjuster:
    def __init__(self, config: RefinementConfig) -> None:
        self.config = config

    def optimize_sparse(self, seed: SparseModel, directory: Path, frame_ids: list[int],
                        package: Path) -> SparseModel:
        """Public sparse-only BA entry point; no feature extraction or dense geometry."""
        require_connected(seed)
        directory.mkdir()
        (directory / "workspace").mkdir()
        seed.write(directory / "initialized_sparse", frame_ids)
        ColmapBinaryWriter().convert(directory / "initialized_sparse", directory / "initialized_binary")
        self._optimize(package, directory)
        validation = ColmapOutputValidator().write_filtered(
            directory / "workspace/ba_raw", directory / "workspace/sparse", directory / "initialized_sparse")
        refined = SparseModel.read(directory / "workspace/sparse", frame_ids)
        write_json(directory / "validation.json", {**validation,
                   "lost_seed_frames": sorted(seed.cameras.keys() - refined.cameras.keys()),
                   "before": model_statistics(seed), "after": model_statistics(refined),
                   "policy": "native standalone BA; Cauchy loss; first camera fixed; no explicit pose/calibration priors"})
        require_connected(refined)
        return refined

    def preflight(self) -> Path:
        spec = importlib.util.find_spec("pycusfm")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("Cannot locate the installed pycusfm package")
        package = Path(next(iter(spec.submodule_search_locations)))
        if not (package / "bin/bundle_adjustment_runner").is_file():
            raise RuntimeError("The installed pyCuSFM package lacks bundle_adjustment_runner")
        return package

    def _optimize(self, package: Path, directory: Path) -> None:
        env = os.environ.copy()
        visible = env.get("CUDA_VISIBLE_DEVICES")
        device = self.config.device
        env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[device] if visible else str(device)
        if env.get("USE_SYSTEM_PROTOBUF", "false").lower() != "true":
            env["LD_LIBRARY_PATH"] = str(package / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        log = directory / "bundle_adjustment.log"
        with Progress("pyCuSFM · Bundle adjustment"), log.open("w") as stream:
            process = subprocess.Popen(bundle_adjustment_command(package, directory), env=env,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = process.wait()
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                except ProcessLookupError:
                    pass
                raise
            stream.flush()
        text = log.read_text(errors="replace").lower()
        if status or "bundle adjustment failed" in text or "ba solve failed" in text:
            raise RuntimeError(f"Sparse BA failed (exit {status}); inspect {log}")

