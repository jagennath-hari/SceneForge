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

"""Sparse map snapshots exported by the native builder for output and dense refinement."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SparseCamera:
    """Pinhole camera in the map's arbitrary-scale world coordinate system."""

    pose: np.ndarray  # [4, 4] camera-to-world; optical axes right, down, forward
    intrinsics: np.ndarray  # [3, 3], in processed-image pixels
    size_hw: tuple[int, int]


@dataclass(frozen=True, slots=True)
class SparsePoint:
    """One landmark and its measured image observations."""

    xyz: np.ndarray  # [3], world coordinates
    rgb: np.ndarray  # [3], uint8 RGB
    observations: dict[int, np.ndarray]  # keyframe index -> [2] pixel coordinates


@dataclass(frozen=True, slots=True)
class SparseModel:
    """Camera and landmark snapshot; optimization state remains in C++."""

    cameras: dict[int, SparseCamera]
    points: tuple[SparsePoint, ...]
