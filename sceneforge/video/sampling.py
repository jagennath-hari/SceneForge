# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jagennath Hari
#
# This file is part of SceneForge.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cached adapter for native streaming keyframe selection and frame extraction."""

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

from sceneforge.utils.validation import require_finite, require_integer
from sceneforge.utils.progress import Progress


@dataclass(frozen=True, slots=True)
class SampledFrames:
    paths: tuple[Path, ...]
    timestamps_seconds: tuple[float, ...]
    source_size: tuple[int, int]
    working_size: tuple[int, int]
    decoder: str
    source_frame_indices: tuple[int, ...]
    candidate_frame_count: int


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
    keyframes: bool = True
    keyframe_config: Path = Path(__file__).resolve().parents[2] / "configs/keyframes_raco.json"

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

    def sample(self, video: Path, destination: Path, on_progress=None) -> SampledFrames:
        if self.keyframes:
            return self._sample_stream(video, destination, on_progress)
        executable = shutil.which("sceneforge-extract-frames")
        if executable is None:
            raise RuntimeError("Native video extractor is missing. Rebuild the Docker environment with "
                               "bash scripts/build_and_start.sh VIDEO_FILE.")
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
                progress.advance(len(cached.paths) - progress.completed)
                document = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
                document["decoded_cache"] = str(cache.resolve())
                progress.status(f"publishing {len(document['frames'])} geometry frames")
                for index, record in enumerate(document["frames"]):
                    source = cache / record["file"]
                    name = f"frame_{index:06d}.png"
                    self._link_or_copy(source, destination / name)
                    record["file"] = name
                (destination / "manifest.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
                if (cache / "parallel_diagnostic.json").is_file():
                    shutil.copy2(cache / "parallel_diagnostic.json", destination / "parallel_diagnostic.json")
        return self._read_manifest(destination)

    def _sample_stream(self, video: Path, destination: Path, on_progress=None) -> SampledFrames:
        executable = shutil.which("sceneforge-select-keyframes")
        if executable is None:
            raise RuntimeError("Native keyframe selector is missing; rebuild Docker")
        video = video.resolve(strict=True)
        config = Path(self.keyframe_config).read_bytes()
        model_args = self.model_arguments(Path(self.keyframe_config))
        model_identity = (Path(model_args[1]) / "manifest.json").read_bytes()
        identity = {
            "version": 2, "mode": "streaming_keyframes", "source": self._source_identity(video),
            "selector": hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
            "config": hashlib.sha256(config).hexdigest(),
            "models": hashlib.sha256(model_identity).hexdigest(),
            "start_seconds": self.start_seconds, "duration": self.duration,
            "count": self.count, "max_edge": self.max_edge,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        root = video.parent / ".keyframes"
        root.mkdir(exist_ok=True)
        cache = root / key
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise ValueError("Frame destination must be empty")
        with Progress("Decoding and selecting keyframes", unit="frame") as progress:
            with (root / f"{key}.lock").open("a") as lock:
                progress.status("waiting for keyframe cache")
                fcntl.flock(lock, fcntl.LOCK_EX)
                cached = self._valid_cache(cache, identity)
                if cached is not None and not (cache / "candidates.json").is_file():
                    cached = None
                reused = cached is not None
                if cached is None:
                    if cache.exists():
                        shutil.rmtree(cache)
                    for abandoned in root.glob(f".{key}-*"):
                        if abandoned.is_dir():
                            shutil.rmtree(abandoned)
                    temporary = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=root))
                    try:
                        def notify(event: dict) -> None:
                            if on_progress is not None:
                                image = (temporary / f"frame_{event['keyframe_index']:06d}.png"
                                         if event.get('event') == 'accepted_keyframe' else None)
                                on_progress(event, image)
                        command = [executable, "--video", str(video), "--output", str(temporary / "manifest.json"),
                                   "--config", str(self.keyframe_config), "--start-seconds", str(self.start_seconds),
                                   "--max-edge", str(self.max_edge), *model_args]
                        if self.duration is not None:
                            command.extend(("--duration", str(self.duration)))
                        if self.count is not None:
                            command.extend(("--frames", str(self.count)))
                        self._run_process(command, progress, "Streaming keyframe selection", notify)
                        document = json.loads((temporary / "manifest.json").read_text())
                        if document.get('keyframe_selection', {}).get('status') != 'complete':
                            raise ValueError("Streaming keyframe selection did not complete")
                        document['source_video'] = identity['source']
                        document['keyframe_selection']['settings'] = json.loads(config)
                        document['keyframe_selection']['model_manifest'] = json.loads(model_identity)
                        (temporary / "manifest.json").write_text(json.dumps(document, indent=2))
                        cached = self._read_manifest(temporary)
                        if self._source_identity(video) != identity['source']:
                            raise RuntimeError("Source video changed during streaming; cache not published")
                        (temporary / "cache.json").write_text(json.dumps(identity))
                        temporary.rename(cache)
                    finally:
                        if temporary.exists():
                            shutil.rmtree(temporary)
                document = json.loads((cache / "manifest.json").read_text())
                document['decoded_cache'] = str(cache.resolve())
                total = document['candidate_frame_count']
                progress.bar.total = total
                progress.advance(total - progress.completed)
                progress.status(f"{'reusing' if reused else 'publishing'} {len(document['frames'])} saved keyframes")
                for index, record in enumerate(document['frames']):
                    source = cache / record['file']
                    self._link_or_copy(source, destination / record['file'])
                    if reused and on_progress is not None:
                        on_progress({'event': 'accepted_keyframe', 'candidate_index': record['candidate_index'],
                                     'keyframe_index': index, 'timestamp_seconds': record['timestamp_seconds']}, source)
                (destination / "manifest.json").write_text(json.dumps(document, indent=2))
                if on_progress is not None:
                    on_progress({'saved': total, 'total': total, 'stage': 'Keyframe selection complete'}, None)
        return self._read_manifest(destination)

    @staticmethod
    def recover_candidates(manifest: dict, decoded: list[dict], indices: list[int], destination: Path) -> dict[int, dict]:
        """Decode requested display timestamps, including codec preroll, saving only requested images."""
        source_identity = manifest['source_video']
        video = Path(source_identity['path'])
        if VideoFrameSampler._source_identity(video) != source_identity:
            raise ValueError('Original video changed; cannot recover intermediate frames')
        executable = shutil.which('sceneforge-extract-frames')
        if executable is None:
            raise RuntimeError('Native frame extractor is missing; rebuild Docker')
        destination.mkdir(parents=True, exist_ok=True)
        result = {}
        # Group nearby requests, so distant weak boundaries do not force a full
        # decode of the intervening video. Every seek decodes codec preroll.
        groups: list[list[int]] = []
        for index in sorted(indices):
            if not groups or decoded[index]['timestamp_seconds'] - decoded[groups[-1][-1]]['timestamp_seconds'] > 2:
                groups.append([])
            groups[-1].append(index)
        with Progress('Recovering intermediate frames', total=len(indices), unit='frame') as progress:
            for group in groups:
                with tempfile.TemporaryDirectory(prefix='.recover-', dir=destination) as name:
                    folder = Path(name)
                    request = folder / 'request.json'
                    request.write_text(json.dumps({'timestamp_origin': manifest['timestamp_origin'],
                                                   'frames': [decoded[index] for index in group]}))
                    output = folder / 'frames'
                    command = [executable, '--input', str(video), '--output', str(output),
                               '--max-edge', str(manifest['max_edge']), '--recover-manifest', str(request)]
                    # Native progress is per group; the parent bar counts recovered frames.
                    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, start_new_session=True)
                    try:
                        if process.wait() != 0:
                            raise RuntimeError('Exact intermediate-frame recovery failed')
                    except BaseException:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                        raise
                    recovered = json.loads((output / 'manifest.json').read_text())
                    records = {record['timestamp_ticks']: record for record in recovered['frames']}
                    for index in group:
                        expected = decoded[index]
                        record = records[expected['timestamp_ticks']]
                        if any(record[key] != expected[key] for key in ('source_frame_index', 'timestamp_seconds')):
                            raise ValueError('Recovered frame identity does not match original decode')
                        retained = destination / f'{index:08d}.png'
                        staged = retained.with_suffix('.partial')
                        shutil.copyfile(output / record['file'], staged)
                        staged.replace(retained)
                        result[index] = {**expected, 'bytes': record['bytes'], 'path': str(retained)}
                        progress.advance()
        if VideoFrameSampler._source_identity(video) != source_identity:
            raise ValueError('Original video changed during frame recovery')
        return result

    @staticmethod
    def model_arguments(config: Path) -> list[str]:
        settings = json.loads(config.read_text(encoding="utf-8"))
        if settings.get("frontend", "raco_aliked_lightglue") != "raco_aliked_lightglue":
            raise ValueError("Only raco_aliked_lightglue keyframes are supported")
        from sceneforge.video.learned_models import LearnedModelCache
        return ["--models", str(LearnedModelCache.from_settings(settings).prepare())]

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
        self._run_process(command, progress, "Native frame extraction")

    @staticmethod
    def _run_process(command: list[str], progress: Progress, operation: str, on_progress=None) -> None:
        # stderr remains visible; stdout contains only the native progress protocol.
        process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, start_new_session=True)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                event = json.loads(line)
                if event.get('event') == 'accepted_keyframe':
                    if on_progress is not None:
                        on_progress(event)
                    continue
                progress.bar.total = event["total"]
                # A hardware failure can restart from frame zero on the CPU.
                if event["saved"] < progress.completed:
                    progress.bar.reset(total=event["total"])
                    progress.completed = 0
                stage = event.get("stage", "saving PNGs")
                timestamp = event.get("timestamp_seconds")
                progress.status(f"video {timestamp:.1f}s | {stage}" if timestamp is not None else stage)
                progress.advance(event["saved"] - progress.completed)
                if on_progress is not None:
                    on_progress(event)
            result = process.wait()
            if result != 0:
                raise RuntimeError(f"{operation} failed (exit {result}); see its error/report above")
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
        if document.get('storage') == 'keyframes_only':
            if document.get('keyframe_selection', {}).get('status') != 'complete' or len(document['frames']) < 3:
                raise ValueError('Incomplete streaming keyframe cache')
            candidate_root = Path(document.get('decoded_cache', directory))
            candidates = json.loads((candidate_root / 'candidates.json').read_text())['frames']
            if len(candidates) != document['candidate_frame_count']:
                raise ValueError('Streaming candidate metadata is incomplete')
            previous_candidate = -1
            for record in document['frames']:
                candidate = record['candidate_index']
                if type(candidate) is not int or not previous_candidate < candidate < document['candidate_frame_count']:
                    raise ValueError('Invalid streaming candidate order')
                if any(record[field] != candidates[candidate][field]
                       for field in ('source_frame_index', 'timestamp_seconds', 'timestamp_ticks')):
                    raise ValueError('Selected keyframe does not match decoded candidate metadata')
                previous_candidate = candidate
        paths, timestamps, source_indices = [], [], []
        for index, record in enumerate(document["frames"]):
            expected = f"frame_{index:06d}.png"
            path = directory / expected
            if record["file"] != expected or not path.is_file() or path.stat().st_size != record["bytes"]:
                raise ValueError("Native manifest contains missing, changed or unordered frame files")
            timestamp = record["timestamp_seconds"]
            if not math.isfinite(timestamp) or (timestamps and timestamp <= timestamps[-1]):
                raise ValueError("Native manifest contains invalid timestamps")
            source_index = record.get("source_frame_index", index)
            if type(source_index) is not int or source_index < 0 or (source_indices and source_index <= source_indices[-1]):
                raise ValueError("Native manifest contains invalid source indices")
            source_indices.append(source_index)
            paths.append(path)
            timestamps.append(timestamp)
        candidate_count = document.get("candidate_frame_count", len(paths))
        if type(candidate_count) is not int or candidate_count < len(paths):
            raise ValueError("Invalid candidate frame count")
        if self.count is not None and candidate_count != self.count:
            raise ValueError("Native extractor returned the wrong frame count")
        source_size = (document["source_width"], document["source_height"])
        working_size = (document["width"], document["height"])
        if any(type(value) is not int or value <= 0 for value in (*source_size, *working_size)):
            raise ValueError("Native manifest contains invalid image dimensions")
        return SampledFrames(tuple(paths), tuple(timestamps), source_size, working_size, document["decoder"],
                             tuple(source_indices), candidate_count)
