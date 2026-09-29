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

"""Image selection and validation before allocating the inference model."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import re

from PIL import Image

from sceneforge.utils.validation import require_integer

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def natural_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    """Tagged parts provide a total ordering even for mixed naming conventions."""
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r"(\d+)", path.name))


def select_image_paths(folder: Path, count: int | None = None) -> tuple[Path, ...]:
    if count is not None:
        require_integer("count", count)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    paths = sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: (natural_key(p), p.name),
    )
    if not paths:
        raise ValueError(f"No supported images found in {folder}")
    if count is not None and len(paths) < count:
        raise ValueError(f"Need {count} images; found {len(paths)} in {folder}")
    return tuple(paths if count is None else paths[:count])


def validate_image_paths(paths: Sequence[Path]) -> tuple[tuple[int, int], ...]:
    if not paths:
        raise ValueError("At least one image is required")
    sizes: list[tuple[int, int]] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                sizes.append((image.height, image.width))
                image.verify()
        except (OSError, SyntaxError) as exc:
            raise ValueError(f"Unreadable or damaged image: {path}") from exc
    if len(set(sizes)) != 1:
        raise ValueError("Use images with identical dimensions from one shot")
    return tuple(sizes)
