"""Subprocess adapter for the native C++ video extractor; no Python decoding."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess

from stereoforge.utils.validation import require_finite, require_integer
from stereoforge.utils.progress import Progress


@dataclass(frozen=True, slots=True)
class SampledFrames:
    paths: tuple[Path, ...]
    timestamps_seconds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class VideoFrameSampler:
    """Invoke native extraction and return its ordered frame manifest."""

    start_seconds: float = 0.0
    duration: float | None = None
    count: int | None = None

    def __post_init__(self) -> None:
        require_finite("start_seconds", self.start_seconds)
        if self.duration is not None:
            require_finite("duration", self.duration)
            if self.duration == 0 or not math.isfinite(self.start_seconds + self.duration):
                raise ValueError("duration must be positive and define a finite end time")
        if self.count is not None:
            require_integer("count", self.count)

    def sample(self, video: Path, destination: Path) -> SampledFrames:
        executable = shutil.which("stereoforge-extract-frames")
        if executable is None:
            raise RuntimeError("Native video extractor is missing. Rebuild the Docker environment with "
                               "bash scripts/build_and_start.sh, then rerun the demo.")
        command = [executable, "--input", str(video), "--output", str(destination),
                   "--start-seconds", str(self.start_seconds)]
        if self.duration is not None:
            command.extend(("--duration", str(self.duration)))
        if self.count is not None:
            command.extend(("--frames", str(self.count)))
        # stderr remains visible; stdout contains only the native progress protocol.
        with Progress("Decoding video", unit="frame") as progress:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, start_new_session=True)
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    event = json.loads(line)
                    progress.bar.total = event["total"]
                    progress.status(f"video {event['timestamp_seconds']:.1f}s | saving PNGs")
                    progress.advance(event["saved"] - progress.completed)
                result = process.wait()
                if result != 0:
                    raise RuntimeError(f"Native frame extraction failed (exit {result}); see its error above")
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
            finally:
                if process.stdout is not None:
                    process.stdout.close()
        document = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if document.get("format_version") != 1 or not document.get("frames"):
            raise ValueError("Invalid native frame manifest")
        paths, timestamps = [], []
        for index, record in enumerate(document["frames"]):
            expected = f"frame_{index:06d}.png"
            if record["file"] != expected or not (destination / expected).is_file():
                raise ValueError("Native manifest contains missing or unordered frame files")
            timestamp = record["timestamp_seconds"]
            if not math.isfinite(timestamp) or (timestamps and timestamp <= timestamps[-1]):
                raise ValueError("Native manifest contains invalid timestamps")
            paths.append(destination / expected)
            timestamps.append(timestamp)
        if self.count is not None and len(paths) != self.count:
            raise ValueError("Native extractor returned the wrong frame count")
        return SampledFrames(tuple(paths), tuple(timestamps))
