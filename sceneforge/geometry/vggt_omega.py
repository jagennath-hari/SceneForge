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

"""Typed inference adapter for the pinned VGGT-Ω package."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import pickle
from typing import Self

import torch
from torch import nn

from sceneforge.utils.camera import invert_world_to_camera
from sceneforge.utils.validation import require_integer

from .config import GeometryConfig
from .inputs import validate_image_paths
from .types import FrameGeometry, GeometrySequence, ImageSize


class GeometryOutOfMemoryError(RuntimeError):
    """An inference allocation failed; the scheduler may retry a smaller section."""


class VGGTOmegaGeometryEstimator:
    """Load a model once, infer an uncut sequence, and return owned CPU tensors.

    Model loading is lazy. Use this as a context manager, or call close() when
    finished. K always describes upstream's processed image grid. Scale remains
    unknown unless meters_per_unit is independently established.
    """

    def __init__(
        self, checkpoint: str | Path, *, device: str = "cuda", image_resolution: int = 512,
        preprocess_mode: str = "balanced", max_frames: int | None = None,
        meters_per_unit: float | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
        self.config = GeometryConfig(
            device=device, image_resolution=image_resolution, preprocess_mode=preprocess_mode,
            max_frames=max_frames, meters_per_unit=meters_per_unit,
        )
        self.device = torch.device(self.config.device)
        self._model: nn.Module | None = None

    @property
    def image_resolution(self) -> int:
        return self.config.image_resolution

    @property
    def preprocess_mode(self) -> str:
        return self.config.preprocess_mode

    @property
    def max_frames(self) -> int | None:
        return self.config.max_frames

    @property
    def meters_per_unit(self) -> float | None:
        return self.config.meters_per_unit

    def load(self) -> None:
        if self._model is not None:
            return
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; check container GPU access")
        from vggt_omega.models import VGGTOmega

        try:
            state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        except (OSError, EOFError, pickle.UnpicklingError, RuntimeError) as exc:
            raise ValueError(
                f"Cannot read checkpoint {self.checkpoint.name}; check that the download is "
                "complete and that this is an official VGGT-Ω state dict."
            ) from exc
        if not isinstance(state, Mapping) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
        ):
            raise ValueError("Expected an official VGGT-Ω state-dict checkpoint")
        # Preserve upstream's working mixed-precision policy for its aggregator.
        model = VGGTOmega(autocast=self.device.type == "cuda")
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "Checkpoint does not match VGGT-Ω. Use a non-text-aligned checkpoint "
                "and its corresponding resolution."
            ) from exc
        del state
        self._model = model.to(self.device).eval()

    def close(self) -> None:
        """Release model references without clearing other users of CUDA's allocator."""
        self._model = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def predict(
        self, image_paths: Sequence[str | Path], start_frame_index: int = 0,
        *, status: Callable[[str], None] | None = None,
    ) -> GeometrySequence:
        paths = tuple(Path(path) for path in image_paths)
        require_integer("start_frame_index", start_frame_index, minimum=0)
        if not paths:
            raise ValueError("Supply at least one image from a continuous recording")
        if self.max_frames is not None and len(paths) > self.max_frames:
            raise ValueError(f"Sequence exceeds configured max_frames={self.max_frames}; no frames were dropped")
        report = status if status is not None else lambda detail: None
        report("validating images")
        sizes = validate_image_paths(paths)
        try:
            report("loading model" if self._model is None else "model ready")
            self.load()
            return self._predict(paths, sizes, start_frame_index, report)
        except torch.cuda.OutOfMemoryError as exc:
            raise GeometryOutOfMemoryError(
                f"GPU memory exhausted while processing {len(paths)} frames in one pass. "
                "A continuous recording can still exceed GPU capacity. Use a shorter clip, "
                "more GPU memory, or explicitly select fewer frames for diagnostics. "
                "No automatic frame dropping or chunking was performed."
            ) from exc

    def _predict(
        self, paths: tuple[Path, ...], sizes: tuple[ImageSize, ...], start: int,
        report: Callable[[str], None],
    ) -> GeometrySequence:
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera

        if self._model is None:
            raise RuntimeError("Model is not loaded")
        with torch.inference_mode():
            report("preprocessing images")
            images = load_and_preprocess_images(
                [str(path) for path in paths], mode=self.preprocess_mode,
                image_resolution=self.image_resolution,
            ).to(self.device)
            report("GPU inference" if self.device.type == "cuda" else "CPU inference")
            predictions = self._model(images)
            report("converting geometry and copying to CPU")
            n, _, h, w = images.shape
            expected_shapes = {
                "images": (1, n, 3, h, w), "pose_enc": (1, n, 9),
                "depth": (1, n, h, w, 1), "depth_conf": (1, n, h, w),
            }
            self._validate_predictions(predictions, expected_shapes)
            extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"], (h, w))
            if extrinsics.shape != (1, n, 3, 4) or intrinsics.shape != (1, n, 3, 3):
                raise ValueError("Unexpected decoded camera shapes from VGGT-Ω")
            poses = invert_world_to_camera(extrinsics[0])
            depth = predictions["depth"][0, ..., 0].float().cpu()
            confidence = predictions["depth_conf"][0].float().cpu()
            intrinsics = intrinsics[0].float().cpu()
            poses = poses.cpu()
            rgb = predictions["images"][0].float().cpu()
            if self.meters_per_unit is not None:
                depth = depth * self.meters_per_unit
                poses[:, :3, 3] *= self.meters_per_unit
        del predictions, images, extrinsics
        # Clone outside inference_mode: consumers receive normal tensors, not
        # inference-only views whose later in-place updates can raise errors.
        depth, confidence, intrinsics, poses, rgb = (
            value.detach().clone() for value in (depth, confidence, intrinsics, poses, rgb)
        )
        frames = tuple(FrameGeometry(start + i, depth[i], confidence[i], intrinsics[i], poses[i])
                       for i in range(len(paths)))
        return GeometrySequence(
            frames=frames, processed_rgb=rgb, source_names=tuple(map(str, paths)),
            original_sizes_hw=sizes,
            units="meters" if self.meters_per_unit is not None else "reconstruction_units",
            meters_per_unit=self.meters_per_unit,
        )

    @staticmethod
    def _validate_predictions(
        predictions: Mapping[str, torch.Tensor], expected_shapes: Mapping[str, tuple[int, ...]],
    ) -> None:
        if not isinstance(predictions, Mapping):
            raise ValueError("VGGT-Ω must return a prediction mapping")
        for key, shape in expected_shapes.items():
            value = predictions.get(key)
            if not isinstance(value, torch.Tensor) or value.shape != shape:
                raise ValueError(f"VGGT-Ω output {key!r} must have shape {shape}")
