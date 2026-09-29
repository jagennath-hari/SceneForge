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

"""Colored point-cloud export, independent of viewer presentation."""
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def write_ply(path: Path, xyz: NDArray, colors: NDArray) -> None:
    xyz = np.asarray(xyz).reshape(-1, 3)
    colors = np.asarray(colors).reshape(-1, 3)
    if len(xyz) != len(colors):
        raise ValueError("Point and color counts differ")
    with path.open("w", encoding="utf-8") as file:
        file.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(xyz, colors, strict=True):
            file.write(f"{point[0]:.6g} {point[1]:.6g} {point[2]:.6g} "
                       f"{color[0]} {color[1]} {color[2]}\n")
