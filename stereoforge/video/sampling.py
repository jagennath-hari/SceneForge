"""Cached subprocess adapter for the native GPU/CPU video extractor."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from stereoforge.utils.validation import require_finite, require_integer
from stereoforge.utils.progress import Progress


@dataclass(frozen=True, slots=True)
class SampledFrames:
    paths: tuple[Path, ...]
    timestamps_seconds: tuple[float, ...]
    source_size: tuple[int, int]
    working_size: tuple[int, int]
    decoder: str


@dataclass(frozen=True, slots=True)
class VideoFrameSampler:
    """Preserve ordered frames, reusing only fully published native extractions.

    Cache identity includes source metadata, selection, working resolution and the
    native executable's contents. Images linked into a run must be treated as
    immutable. The original video remains the source for full-resolution output.
    """

    start_seconds: float = 0.0
    duration: float | None = None
    count: int | None = None
    max_edge: int = 1536

    def __post_init__(self) -> None:
        require_finite("start_seconds", self.start_seconds)
        if self.duration is not None:
            require_finite("duration", self.duration)
            if self.duration == 0 or not math.isfinite(self.start_seconds + self.duration):
                raise ValueError("duration must be positive and define a finite end time")
        if self.count is not None:
            require_integer("count", self.count)
        require_integer("max_edge", self.max_edge, minimum=0)
        if self.max_edge == 1:
            raise ValueError("max_edge must be zero or at least two")

    def sample(self, video: Path, destination: Path) -> SampledFrames:
        executable = shutil.which("stereoforge-extract-frames")
        if executable is None:
            raise RuntimeError("Native video extractor is missing. Rebuild the Docker environment with "
                               "bash scripts/build_and_start.sh, then rerun the demo.")
        video = video.resolve(strict=True)
        identity = {
            "version": 1, "source": self._source_identity(video),
            "extractor": hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
            "start_seconds": self.start_seconds, "duration": self.duration,
            "count": self.count, "max_edge": self.max_edge,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache_root = video.parent / ".frames"
        cache_root.mkdir(parents=True, exist_ok=True)
        cache = cache_root / key
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise ValueError("Frame destination must be empty")
        with Progress("Decoding video", unit="frame") as progress:
            progress.status("checking frame cache")
            # Serialize publication for the same source/options. Other videos can
            # still extract concurrently, and a killed process releases its lock.
            with (cache_root / f"{key}.lock").open("a") as lock:
                progress.status("waiting for frame cache")
                fcntl.flock(lock, fcntl.LOCK_EX)
                cached = self._valid_cache(cache, identity)
                if cached is None:
                    if cache.exists():
                        shutil.rmtree(cache)
                    # A hard kill may leave an unpublished directory. The lock
                    # proves no other extractor for this key is using it.
                    for abandoned in cache_root.glob(f".{key}-*"):
                        if abandoned.is_dir():
                            shutil.rmtree(abandoned)
                    temporary = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=cache_root))
                    try:
                        self._extract(executable, video, temporary, progress)
                        cached = self._read_manifest(temporary)
                        if self._source_identity(video) != identity["source"]:
                            raise RuntimeError("Source video changed during extraction; cache was not published")
                        (temporary / "cache.json").write_text(json.dumps(identity), encoding="utf-8")
                        temporary.rename(cache)
                    finally:
                        if temporary.exists():
                            shutil.rmtree(temporary)
                progress.bar.total = len(cached.paths)
                progress.status("reusing saved frames" if progress.completed == 0 else "publishing saved frames")
                for path in cached.paths:
                    self._link_or_copy(cache / path.name, destination / path.name)
                shutil.copy2(cache / "manifest.json", destination / "manifest.json")
                if (cache / "parallel_diagnostic.json").is_file():
                    shutil.copy2(cache / "parallel_diagnostic.json", destination / "parallel_diagnostic.json")
                progress.advance(len(cached.paths) - progress.completed)
        return self._read_manifest(destination)

    @staticmethod
    def _source_identity(video: Path) -> dict[str, str | int]:
        stat = video.stat()
        return {"path": str(video), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns, "inode": stat.st_ino}

    @staticmethod
    def _link_or_copy(source: Path, destination: Path) -> None:
        try:
            os.link(source, destination)
        except OSError as error:
            if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
                raise
            shutil.copy2(source, destination)

    def _valid_cache(self, cache: Path, identity: dict) -> SampledFrames | None:
        try:
            if json.loads((cache / "cache.json").read_text(encoding="utf-8")) != identity:
                return None
            return self._read_manifest(cache)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _extract(self, executable: str, video: Path, destination: Path, progress: Progress) -> None:
        command = [executable, "--input", str(video), "--output", str(destination),
                   "--start-seconds", str(self.start_seconds), "--max-edge", str(self.max_edge)]
        if self.duration is not None:
            command.extend(("--duration", str(self.duration)))
        if self.count is not None:
            command.extend(("--frames", str(self.count)))
        # stderr remains visible; stdout contains only the native progress protocol.
        process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, start_new_session=True)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                event = json.loads(line)
                progress.bar.total = event["total"]
                # A hardware failure can restart from frame zero on the CPU.
                if event["saved"] < progress.completed:
                    progress.bar.reset(total=event["total"])
                    progress.completed = 0
                progress.status(f"video {event['timestamp_seconds']:.1f}s | {event.get('stage', 'saving PNGs')}")
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

    def _read_manifest(self, directory: Path) -> SampledFrames:
        document = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if document.get("format_version") != 1 or not document.get("frames"):
            raise ValueError("Invalid native frame manifest")
        if document["max_edge"] != self.max_edge:
            raise ValueError("Frame cache resolution does not match the request")
        paths, timestamps = [], []
        for index, record in enumerate(document["frames"]):
            expected = f"frame_{index:06d}.png"
            path = directory / expected
            if record["file"] != expected or not path.is_file() or path.stat().st_size != record["bytes"]:
                raise ValueError("Native manifest contains missing, changed or unordered frame files")
            timestamp = record["timestamp_seconds"]
            if not math.isfinite(timestamp) or (timestamps and timestamp <= timestamps[-1]):
                raise ValueError("Native manifest contains invalid timestamps")
            paths.append(path)
            timestamps.append(timestamp)
        if self.count is not None and len(paths) != self.count:
            raise ValueError("Native extractor returned the wrong frame count")
        source_size = (document["source_width"], document["source_height"])
        working_size = (document["width"], document["height"])
        if any(type(value) is not int or value <= 0 for value in (*source_size, *working_size)):
            raise ValueError("Native manifest contains invalid image dimensions")
        return SampledFrames(tuple(paths), tuple(timestamps), source_size, working_size, document["decoder"])
