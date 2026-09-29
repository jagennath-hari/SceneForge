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
#include <cstdint>

namespace sceneforge::video {
inline constexpr int ransac_hypotheses = 2000;
inline constexpr int ransac_models = 2 * ransac_hypotheses + 2;
inline constexpr int maximum_keypoints = 4096;
struct DeviceMatch final {
    float ax, ay, bx, by;
    int source, target;
    float confidence;
};
struct RansacModel final {
    double matrix[9];
    int inliers;
    float cost;
    int homography;
    int valid;
};
struct RansacSummary final {
    int matches, inliers, homography, invalid;
    float coverage, displacement;
};
// All pointers except dimensions/settings refer to caller-owned device memory.
// Enqueued on the matcher stream; no host readback or allocation between stages.
void verifyDeviceMatches(const void* points0, const void* scores0, const void* points1,
    const void* scores1, const std::int32_t* indices, const void* confidence, bool half,
    int count, int width0, int height0, int width1, int height1,
    float detector_threshold, float match_threshold, float pixel_threshold,
    unsigned seed, DeviceMatch* matches, int* match_count, RansacModel* models,
    int* winners, unsigned char* mask, RansacSummary* summary, cudaStream_t stream);
}  // namespace sceneforge::video
