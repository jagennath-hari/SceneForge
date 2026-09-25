"""Tensor loading and final export for the in-process C++ common-map builder."""

import importlib
import json
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
    if module.api_version != 1:
        raise RuntimeError('Native common-map API mismatch; rebuild Docker')
    return module


class NativeMap:
    def __init__(self, device: int, iterations: int, diagnostics: bool) -> None:
        self.backend = native_backend()
        self.builder = self.backend.MapBuilder(device, iterations, diagnostics)
        self.stage_seconds: dict[str, float] = {}

    def load_tracks(self, path: Path) -> None:
        with path.open() as stream:
            tracks = [{int(frame): np.asarray(uv, dtype=np.float64)
                       for frame, uv in json.loads(line).items()} for line in stream]
        self.builder.set_tracks(tracks)

    def add_window(self, path: Path, images: list[Path], progress) -> None:
        sequence = load_sequence(path)
        if sequence.processed_rgb is None:
            raise ValueError('Window has no processed RGB for feature-grid verification')
        frames = []
        for index, frame in enumerate(sequence.frames):
            rgb = np.ascontiguousarray((sequence.processed_rgb[index].numpy().transpose(1, 2, 0)*255).round(), dtype=np.uint8)
            with Image.open(images[frame.frame_index]) as image:
                if not np.array_equal(np.asarray(image.convert('RGB')), rgb):
                    raise ValueError(f'Feature/VGGT image grid mismatch at frame {frame.frame_index}')
            frames.append(self.backend.DepthFrame(frame.frame_index,
                frame.camera_to_world.numpy().astype(np.float64), frame.intrinsics.numpy().astype(np.float64),
                np.ascontiguousarray(frame.depth.numpy(), dtype=np.float32), rgb))
        stage = None
        started = perf_counter()

        def update(name: str) -> None:
            nonlocal stage, started
            now = perf_counter()
            if stage is not None:
                self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + now - started
            stage, started = name, now
            progress(name)

        try:
            self.builder.add_window(frames, update)
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
