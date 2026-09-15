"""Standalone interactive ORB/RANSAC selection; no geometry models are loaded."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from stereoforge.utils.progress import Progress
from stereoforge.video.sampling import VideoFrameSampler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Decode or reuse the full video frame cache")
    source.add_argument("--input", type=Path, help="Existing decoded candidate directory with manifest.json")
    parser.add_argument("--output", type=Path, help="New diagnostic directory")
    parser.add_argument("--keyframe-config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "configs/keyframes.json")
    args = parser.parse_args()
    try:
        executable = shutil.which("stereoforge-select-keyframes")
        if executable is None:
            raise RuntimeError("Native selector is missing; stop the container and rebuild Docker")
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("The debug window needs a desktop display and Docker X11 forwarding")
        config = args.keyframe_config.resolve(strict=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output = (args.output or Path("data/intermediate") / f"keyframes_{stamp}").resolve()
        output.mkdir(parents=True, exist_ok=False)
        if args.video is not None:
            candidates = output / "candidates"
            VideoFrameSampler(keyframes=False).sample(args.video, candidates)
        else:
            candidates = args.input.resolve(strict=True)
        report = output / "selection.json"
        # Keep diagnostics separate from the production selection cache. Always
        # rerun selection, even when decoding and a prior selection are cached.
        with Progress("Keyframe debug", unit="frame") as progress:
            VideoFrameSampler._run_process(
                [executable, "--input", str(candidates), "--output", str(report),
                 "--config", str(config), "--debug-view"], progress, "Keyframe debug")
        document = json.loads(report.read_text(encoding="utf-8"))
        document["candidate_directory"] = str(candidates)
        document["keyframe_selection"]["settings"] = json.loads(config.read_text(encoding="utf-8"))
        report.write_text(json.dumps(document, indent=2), encoding="utf-8")
        print(f"Status: {document['keyframe_selection']['status']}; "
              f"{len(document['frames'])} keyframes. Report: {report}")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Stopped. Use Q in the window to save a partial selection report.\n")


if __name__ == "__main__":
    raise SystemExit(main())
