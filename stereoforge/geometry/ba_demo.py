"""Retry only bundle adjustment using a run's saved depth-initialized landmarks."""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess

import torch

from stereoforge.utils.artifacts import write_json
from stereoforge.utils.progress import Progress
from stereoforge.refinement.colmap_binary import ColmapBinaryWriter, bundle_adjustment_command
from stereoforge.refinement.colmap_validation import ColmapOutputValidator
from stereoforge.refinement.pycusfm import CuSFMRefiner
from stereoforge.refinement.report import write_refinement_report, write_refinement_status

from .config import RefinementConfig


def publish_saved_ba(source: Path, output: Path, metadata: dict) -> int:
    """Revalidate completed BA output into a new report without native execution."""
    output.mkdir()
    (output / "workspace").mkdir()
    with Progress("Preparing saved optimized output"):
        shutil.copytree(source / "initialized_sparse", output / "initialized_sparse")
        shutil.copytree(source / "workspace/ba_raw", output / "workspace/ba_raw")
        for name in ("bundle_adjustment.log", "initialization.json"):
            if (source / name).is_file():
                shutil.copy2(source / name, output / name)
    with Progress("Validating optimized landmarks"):
        validation = ColmapOutputValidator().write_filtered(
            output / "workspace/ba_raw", output / "workspace/sparse", output / "initialized_sparse")
    write_json(output / "refinement.json", {
        "feature_type": "aliked", "source_ba_result": str(source), "validation": validation,
        "landmark_initialization": "vggt_depth", "dense_depth_refined": False,
        "bundle_adjustment_backend": "bundle_adjustment_runner", "report_only": True,
    })
    summary = write_refinement_report(output, metadata, max_points=240000)
    logging.info("%s: %d/%d frames, %d points. Open %s", summary.get("status"),
                 summary.get("registered_frames", 0), len(metadata["frames"]),
                 summary.get("sparse_points", 0), output / "index.html")
    return 0 if summary.get("status") == "complete" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Saved full geometry run")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ba-result", type=Path,
                        help="Rebuild only the report from a completed BA directory; skip optimization")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run = args.run.expanduser().resolve()
    output = run / datetime.now(timezone.utc).strftime("pycusfm_ba_%Y%m%d_%H%M%S_%f")
    try:
        metadata = json.loads((run / "metadata.json").read_text())
        if args.ba_result is not None:
            source_ba = args.ba_result.expanduser().resolve()
            if source_ba.parent != run or not source_ba.name.startswith("pycusfm_ba_"):
                raise ValueError("--ba-result must be a pycusfm_ba_* retry directory inside --run")
            return publish_saved_ba(source_ba, output, metadata)
        config = RefinementConfig(device=args.device)
        source = run / "pycusfm"
        if not (source / "initialized_sparse/points3D.txt").is_file():
            raise ValueError("This run has no saved pycusfm/initialized_sparse landmarks to retry")
        package = CuSFMRefiner(config).preflight()
        if args.device >= torch.cuda.device_count():
            raise ValueError("--device is outside visible CUDA devices")
        output.mkdir()
        (output / "workspace").mkdir()
        with Progress("Preparing saved BA input"):
            shutil.copytree(source / "initialized_sparse", output / "initialized_sparse")
            ColmapBinaryWriter().convert(output / "initialized_sparse", output / "initialized_binary")
        initialization_path = source / "initialization.json"
        initialization = json.loads(initialization_path.read_text()) if initialization_path.is_file() else {}
        write_json(output / "initialization.json", initialization)
        env = os.environ.copy()
        visible = env.get("CUDA_VISIBLE_DEVICES")
        env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[args.device] if visible else str(args.device)
        if env.get("USE_SYSTEM_PROTOBUF", "false").lower() != "true":
            env["LD_LIBRARY_PATH"] = str(package / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        log = output / "bundle_adjustment.log"
        with Progress("pyCuSFM · Depth-initialized bundle adjustment"), log.open("w") as stream:
            process = subprocess.Popen(bundle_adjustment_command(package, output), env=env,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = process.wait()
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            stream.flush()
            text = log.read_text(errors="replace")
            if status or "bundle adjustment failed" in text.lower() or "BA solve failed" in text:
                raise RuntimeError(f"Bundle adjustment failed (exit {status}); inspect {log}\n"
                                   + "\n".join(text.splitlines()[:20]))
        with Progress("Validating optimized landmarks"):
            validation = ColmapOutputValidator().write_filtered(
                output / "workspace/ba_raw", output / "workspace/sparse", output / "initialized_sparse")
        write_json(output / "refinement.json", {
            "feature_type": "aliked", "stages": ["bundle_adjustment"], "source_run": str(run),
            "landmark_initialization": "vggt_depth", "initialization": initialization,
            "validation": validation, "dense_depth_refined": False,
            "bundle_adjustment_backend": "bundle_adjustment_runner", "input_format": "colmap_binary",
        })
        summary = write_refinement_report(output, metadata, max_points=240000)
        logging.info("%s: %d/%d frames, %d points. Open %s", summary.get("status"),
                     summary.get("registered_frames", 0), len(metadata["frames"]),
                     summary.get("sparse_points", 0), output / "index.html")
        return 0 if summary.get("status") == "complete" else 1
    except (Exception, KeyboardInterrupt) as exc:
        message = str(exc) or "Interrupted"
        write_refinement_status(output, message)
        logging.error("%s\nDiagnostic files retained in %s", message, output)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
