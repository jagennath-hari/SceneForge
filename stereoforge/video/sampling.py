"""Decode an uncut recording, optionally selecting an explicit diagnostic segment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from stereoforge.utils.validation import require_finite, require_integer
from stereoforge.utils.progress import Progress


@dataclass(frozen=True, slots=True)
class SampledFrames:
    paths: tuple[Path, ...]
    timestamps_seconds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class VideoFrameSampler:
    """Default: every decoded frame, in order, until EOF.

    A count alone selects the first N frames. A duration alone selects every
    frame within that segment. Count plus duration explicitly samples N frames.
    No frames are implicitly discarded or divided into inference windows.
    """

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
        """Write RGB PNGs and preserve actual timestamps; decoding errors fail the run."""
        import av

        if not video.is_file():
            raise FileNotFoundError(f"Video not found: {video}")
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise ValueError(f"Frame destination must be empty: {destination}")
        paths: list[Path] = []
        timestamps: list[float] = []
        targets = None
        if self.count is not None and self.duration is not None:
            targets = [self.start_seconds + i * self.duration / self.count for i in range(self.count)]
        end_seconds = self.start_seconds + self.duration if self.duration is not None else None
        try:
            with av.open(str(video)) as container, Progress("Decoding video", unit="frame") as progress:
                if not container.streams.video:
                    raise ValueError("Input has no video stream")
                stream = container.streams.video[0]
                # Container counts can be missing or inaccurate. The total is
                # display metadata only and never controls how much is decoded.
                progress.bar.total = self.count or (
                    stream.frames if not self.start_seconds and self.duration is None and stream.frames > 0 else None
                )
                progress.status("decoding and saving PNGs")
                if stream.time_base is None or stream.time_base <= 0:
                    raise ValueError("Video stream has no valid time base")
                origin = stream.start_time
                # No seek for a complete recording: begin at the first decoded frame.
                if self.start_seconds > 0:
                    container.seek(
                        (origin or 0) + int(self.start_seconds / float(stream.time_base)),
                        stream=stream, backward=True, any_frame=False,
                    )
                previous_timestamp: float | None = None
                origin_seconds = float(origin * stream.time_base) if origin is not None else None
                for frame in container.decode(stream):
                    if frame.pts is None or frame.time_base is None:
                        raise ValueError("Video contains a frame without a timestamp; cannot preserve timing")
                    absolute_seconds = float(frame.pts * frame.time_base)
                    if origin_seconds is None:
                        if self.start_seconds > 0:
                            raise ValueError("Video lacks a start timestamp; decode the complete clip without --start-seconds")
                        origin_seconds = absolute_seconds
                    timestamp = absolute_seconds - origin_seconds
                    if previous_timestamp is not None and timestamp <= previous_timestamp:
                        raise ValueError("Video contains non-increasing timestamps; no frames were silently discarded")
                    previous_timestamp = timestamp
                    # For a full decode, keep every frame even if metadata puts the
                    # first timestamp slightly before stream.start_time.
                    if self.start_seconds > 0 and timestamp < self.start_seconds:
                        continue
                    if end_seconds is not None and timestamp >= end_seconds:
                        break
                    if targets is not None and timestamp < targets[len(paths)]:
                        continue
                    path = destination / f"frame_{len(paths):06d}.png"
                    frame.to_image().convert("RGB").save(path)
                    paths.append(path)
                    timestamps.append(timestamp)
                    progress.status(f"video {timestamp:.1f}s | saving PNGs")
                    progress.advance()
                    if self.count is not None and len(paths) == self.count:
                        break
        except av.FFmpegError as exc:
            raise ValueError(f"Could not decode {video}: {exc}") from exc
        if not paths:
            raise ValueError("No frames decoded; check the video and requested start time")
        if self.count is not None and len(paths) != self.count:
            raise ValueError(
                f"Extracted {len(paths)}/{self.count} frames. Choose a longer segment "
                "or fewer frames; check that the segment is before the end of the video."
            )
        return SampledFrames(tuple(paths), tuple(timestamps))
