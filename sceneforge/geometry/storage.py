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

"""CPU tensor artifacts for VGGT-Ω windows."""
from pathlib import Path
import torch
from .types import FrameGeometry, GeometrySequence

def save_sequence(sequence: GeometrySequence, path: Path) -> None:
    # Tensor-only dictionaries can be loaded with weights_only=True.
    torch.save({"frames": [dict(frame_index=f.frame_index, depth=f.depth.clone(),
                               confidence=f.confidence.clone(), intrinsics=f.intrinsics.clone(),
                               camera_to_world=f.camera_to_world.clone()) for f in sequence.frames],
                "rgb": sequence.processed_rgb, "names": sequence.source_names,
                "sizes": sequence.original_sizes_hw}, path)


def load_sequence(path: Path) -> GeometrySequence:
    data = torch.load(path, map_location="cpu", weights_only=True)
    return GeometrySequence(tuple(FrameGeometry(**f) for f in data["frames"]),
                            data["rgb"], data["names"], data["sizes"])

