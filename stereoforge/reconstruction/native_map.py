"""Tensor loading and final export for the in-process C++ common-map builder."""

import importlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from stereoforge.geometry.storage import load_sequence
from stereoforge.refinement.sparse_model import SparseCamera, SparseModel, SparsePoint


def native_backend():
    try:
        module = importlib.import_module('_stereoforge_map')
    except ImportError as error:
        raise RuntimeError('Rebuild Docker to install the native common-map builder (_stereoforge_map)') from error
    if getattr(module, "api_version", None) != 7:
        raise RuntimeError('Native common-map module is outdated. Stop the old container, rebuild Docker, and start a new container.')
    return module


class WindowRejected(RuntimeError):
    """Native candidate rejection; the accepted map is unchanged."""


class NativeMap:
    def __init__(self, device: int, iterations: int, diagnostics: bool) -> None:
        self.backend = native_backend()
        self.builder = self.backend.MapBuilder(device, iterations, diagnostics)
        self.stage_seconds: dict[str, float] = {}
        self.last_stage: str | None = None
        self.alignment_warnings: list[str] = []
        self._frame_cache: OrderedDict[Path, dict] = OrderedDict()

    def load_tracks(self, path: Path) -> None:
        with path.open() as stream:
            tracks = [{int(frame): np.asarray(uv, dtype=np.float64)
                       for frame, uv in json.loads(line).items()} for line in stream]
        self.builder.set_tracks(tracks)

    def _load_frames(self, path: Path, images: list[Path]) -> dict:
        if path in self._frame_cache:
            self._frame_cache.move_to_end(path)
            return self._frame_cache[path]
        sequence = load_sequence(path)
        if sequence.processed_rgb is None:
            raise ValueError('Window has no processed RGB for feature-grid verification')
        frames = {}
        for index, frame in enumerate(sequence.frames):
            rgb = np.ascontiguousarray((sequence.processed_rgb[index].numpy().transpose(1, 2, 0)*255).round(), dtype=np.uint8)
            with Image.open(images[frame.frame_index]) as image:
                if not np.array_equal(np.asarray(image.convert('RGB')), rgb):
                    raise ValueError(f'Feature/VGGT image grid mismatch at frame {frame.frame_index}')
            frames[frame.frame_index] = self.backend.DepthFrame(frame.frame_index,
                frame.camera_to_world.numpy().astype(np.float64), frame.intrinsics.numpy().astype(np.float64),
                np.ascontiguousarray(frame.depth.numpy(), dtype=np.float32), rgb)
        self._frame_cache[path] = frames
        while len(self._frame_cache) > 2:
            self._frame_cache.popitem(last=False)
        return frames

    def add_window(self, path: Path, images: list[Path], references: dict[int, Path], progress) -> None:
        frames = self._load_frames(path, images)
        reference_frames = []
        for source in sorted(set(references.values())):
            saved = self._load_frames(source, images)
            reference_frames.extend(saved[frame] for frame, owner in references.items() if owner == source)
        try:
            self._invoke(lambda update: self.builder.add_window(list(frames.values()), reference_frames, update), progress)
        except RuntimeError as error:
            raise WindowRejected(str(error)) from error

    def rank_windows(self, windows: list[list[int]]) -> list[int]:
        return self.builder.rank_windows(windows)

    def finalize(self, progress) -> None:
        self._invoke(self.builder.finalize, progress)

    def _invoke(self, operation, progress) -> None:
        stage = None
        started = perf_counter()

        def update(name: str) -> None:
            nonlocal stage, started
            if name.startswith('WARNING: '):
                message = name.removeprefix('WARNING: ')
                if message not in self.alignment_warnings:
                    self.alignment_warnings.append(message)
                    logging.getLogger(__name__).warning('%s', message)
                return
            now = perf_counter()
            if stage is not None:
                self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + now - started
            stage, started = name, now
            self.last_stage = name
            progress(name)

        try:
            operation(update)
        finally:
            if stage is not None:
                self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + perf_counter() - started

    @property
    def accepted_windows(self) -> int:
        return self.builder.accepted_windows

    def export(self) -> SparseModel | None:
        if not self.accepted_windows:
            return None
        model = self.builder.map
        cameras = {}
        for frame, camera in model.cameras.items():
            pose = np.eye(4)
            pose[:3, :3] = camera.rotation
            pose[:3, 3] = camera.center
            cameras[frame] = SparseCamera(pose, camera.intrinsics, (camera.height, camera.width))
        points = tuple(SparsePoint(point.position, np.asarray(point.color, dtype=np.uint8), point.observations)
                       for point in model.landmarks.values())
        return SparseModel(cameras, points)
