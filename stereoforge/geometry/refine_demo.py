"""Diagnose pyCuSFM on saved VGGT frames without decoding or running VGGT again."""

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm.contrib.logging import logging_redirect_tqdm

from stereoforge.refinement.pycusfm import CuSFMRefiner
from stereoforge.refinement.report import write_refinement_report, write_refinement_status

from .config import DemoConfig
from .types import FrameGeometry, GeometrySequence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=16, help="Number of saved frames for diagnosis (default: 16)")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Spacing between diagnostic frames; increase to cover more camera motion (default: 1)")
    parser.add_argument("--debug", action="store_true", help="Save upstream feature/match visualization artifacts")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/default.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.frames < 3:
        parser.error("--frames must be at least 3")
    if args.frame_step < 1:
        parser.error("--frame-step must be positive")
    run = args.run.expanduser().resolve()
    output = run / datetime.now(timezone.utc).strftime("pycusfm_diagnostic_%Y%m%d_%H%M%S_%f")
    try:
        config = DemoConfig.from_yaml(args.config)
        metadata = json.loads((run / "metadata.json").read_text())
        metadata = deepcopy(metadata)
        indices = list(range(0, len(metadata["frames"]), args.frame_step))[:args.frames]
        if len(indices) < 3:
            raise ValueError("Frame selection leaves fewer than three saved frames; reduce --frame-step")
        metadata["frames"] = [metadata["frames"][i] for i in indices]
        count = len(metadata["frames"])
        timestamps = metadata.get("provenance", {}).get("video_timestamps_seconds")
        if timestamps is not None:
            timestamps = tuple(timestamps[i] for i in indices)
            metadata["provenance"]["video_timestamps_seconds"] = timestamps
        metadata.setdefault("provenance", {})["diagnostic_source_frame_indices"] = indices
        logging.info("Refining %d saved frames, source indices %d–%d (step %d)",
                     count, indices[0], indices[-1], args.frame_step)
        with np.load(run / "geometry.npz", allow_pickle=False) as arrays:
            depth, confidence = arrays["depth"][indices], arrays["confidence"][indices]
            intrinsics, poses = arrays["intrinsics"][indices], arrays["camera_to_world"][indices]
            rgb = arrays["processed_rgb"][indices]
        sequence = GeometrySequence(
            tuple(FrameGeometry(i, torch.from_numpy(depth[i]), torch.from_numpy(confidence[i]),
                                torch.from_numpy(intrinsics[i]), torch.from_numpy(poses[i])) for i in range(count)),
            torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255,
            tuple(f["source"] for f in metadata["frames"]),
            tuple(tuple(f["original_size_hw"]) for f in metadata["frames"]),
            metadata["units"], metadata.get("meters_per_unit"),
        )
        with logging_redirect_tqdm():
            CuSFMRefiner(replace(config.refinement, debug=args.debug)).run(sequence, timestamps, output)
            summary = write_refinement_report(output, metadata, config.preview.max_points)
        logging.info("Diagnostic status: %s. Open %s", summary.get("status"), output / "index.html")
        return 0 if summary.get("status") == "complete" else 1
    except (Exception, KeyboardInterrupt) as exc:
        message = str(exc) or "Interrupted"
        write_refinement_status(output, message)
        logging.error("%s\nDiagnostic files retained in %s", message, output)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
