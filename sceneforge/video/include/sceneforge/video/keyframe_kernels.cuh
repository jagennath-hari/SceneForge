// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Jagennath Hari
//
// This file is part of SceneForge.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>

namespace sceneforge::video {
// Antialiased Keys bicubic (a=-0.5), half-pixel centers and clamped edges.
// Caller owns buffers and orders upload -> preprocessing -> inference on stream.
// Intermediate workspace: 3 * width * source_height floats, RGB planar.
void prepareRgbImage(const unsigned char* pixels, int source_width, int source_height,
    void* tensor, bool half, int width, int height, int canvas_width, int canvas_height,
    float* intermediate, std::size_t intermediate_bytes, float* statistics, cudaStream_t stream);
void validateDescriptors(const void* descriptors, bool half, std::size_t count,
    float* statistics, cudaStream_t stream);
void validateKeypoints(const void* points, const void* scores, bool half, int count,
    int width, int height, float detector_threshold, float* statistics, cudaStream_t stream);
}  // namespace sceneforge::video
