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

"""Camera-convention conversion at third-party integration boundaries."""

from __future__ import annotations

import torch


def invert_world_to_camera(extrinsics: torch.Tensor) -> torch.Tensor:
    """Invert [N,3,4] rigid extrinsics using R.T and -R.T @ t."""
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (3, 4):
        raise ValueError("Expected world-to-camera extrinsics [N,3,4]")
    if not extrinsics.is_floating_point() or not torch.isfinite(extrinsics).all():
        raise ValueError("Extrinsics must be finite floating-point tensors")
    source = extrinsics.float()
    rotation = source[:, :3, :3]
    identity = torch.eye(3, device=source.device).expand(len(source), 3, 3)
    if not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-3, rtol=0):
        raise ValueError("Extrinsics contain a non-rigid rotation")
    if not torch.allclose(torch.linalg.det(rotation), source.new_ones(len(source)), atol=1e-3, rtol=0):
        raise ValueError("Extrinsic rotations must have determinant +1")
    poses = torch.eye(4, device=source.device).repeat(len(source), 1, 1)
    poses[:, :3, :3] = rotation.transpose(-1, -2)
    poses[:, :3, 3] = -(rotation.transpose(-1, -2) @ source[:, :3, 3:]).squeeze(-1)
    return poses
