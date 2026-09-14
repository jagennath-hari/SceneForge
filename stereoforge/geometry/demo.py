"""Reconstruct every frame with adaptive GPU sections and export a geometry report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
import logging
from pathlib import Path
from tqdm.contrib.logging import logging_redirect_tqdm

from .config import DemoConfig
from .runner import DemoRequest, GeometryDemoRunner

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--images", type=Path, help="Naturally sorted images from ONE shot")
    source.add_argument("--video", type=Path, help="Continuous video; defaults to every frame through EOF")
    parser.add_argument("--start-seconds", type=float, help="Optional diagnostic start time; defaults to beginning")
    parser.add_argument("--duration", type=float, help="Optional diagnostic segment duration; default is through EOF")
    parser.add_argument("--frames", type=int, help="Optional diagnostic frame count; default is all frames")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--checkpoint", type=Path, help="Explicit authorized local checkpoint; otherwise use the Hugging Face cache or gated download")
    # Accept old commands without requiring a separate download mode.
    parser.add_argument("--download-checkpoint", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help="New/empty output folder; defaults to a timestamped run")
    parser.add_argument("--device", help="cuda uses all visible GPUs; cuda:N selects one; cpu uses CPU")
    parser.add_argument("--resolution", type=int, help="Match checkpoint resolution (default: 512)")
    parser.add_argument("--meters-per-unit", type=float, help="Known scale only; scales depth AND camera translations")
    parser.add_argument("--debug", action="store_true", help="Include tracebacks for unexpected failures")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("dinov3").setLevel(logging.DEBUG if args.debug else logging.WARNING)
    try:
        config = DemoConfig.from_yaml(args.config)
        config = replace(config, geometry=replace(
            config.geometry,
            device=args.device if args.device is not None else config.geometry.device,
            image_resolution=args.resolution if args.resolution is not None else config.geometry.image_resolution,
            meters_per_unit=(args.meters_per_unit if args.meters_per_unit is not None
                             else config.geometry.meters_per_unit),
        ))
        timestamp = datetime.now(timezone.utc).strftime("geometry_%Y%m%d_%H%M%S_%f")
        request = DemoRequest(
            output=args.output or ROOT / "data/intermediate" / timestamp,
            checkpoint=args.checkpoint,
            images=None if args.video else args.images or ROOT / "data/input/frames",
            video=args.video,
            frame_count=args.frames, start_seconds=args.start_seconds, duration=args.duration,
        )
        with logging_redirect_tqdm():
            result = GeometryDemoRunner(config).run(request)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; incomplete report files were not published")
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        if args.debug:
            LOGGER.exception("Geometry demo failed")
        else:
            LOGGER.error("%s (use --debug for a traceback)", exc)
        return 1
    for frame in result.frames:
        LOGGER.debug("Frame %d: valid=%.1f%%, median depth=%s %s",
                    frame.frame_index, frame.valid_fraction * 100, frame.median_depth, result.units)
    LOGGER.info("Saved geometry and previews to %s", result.output)
    LOGGER.info("Open index.html in the corresponding host data/intermediate directory")
    if not any(frame.valid_fraction > 0 for frame in result.frames):
        LOGGER.warning("No valid geometry; inspect the RGB/confidence previews and checkpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
