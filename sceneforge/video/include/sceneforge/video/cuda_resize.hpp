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

#include "sceneforge/video/ffmpeg_resources.hpp"

namespace sceneforge::video::detail {

// Resize supported NVDEC surfaces in their owning CUDA context, then download.
// Other formats are downloaded unchanged for the CPU area resizer.
class CudaResizer final {
public:
    [[nodiscard]] FramePtr download(const AVFrame& source, int width, int height);
private:
    BufferPtr pool_;
};

}  // namespace sceneforge::video::detail
