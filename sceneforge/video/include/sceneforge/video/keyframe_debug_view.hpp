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

#include "sceneforge/video/keyframe_selector.hpp"
#include <string>
#include <vector>

namespace sceneforge::video {
class KeyframeDebugView final {
public:
    explicit KeyframeDebugView(const std::filesystem::path& directory);
    ~KeyframeDebugView();
    KeyframeDebugView(const KeyframeDebugView&) = delete;
    KeyframeDebugView& operator=(const KeyframeDebugView&) = delete;
    [[nodiscard]] bool show(const FrameFeatures* reference, const FrameFeatures& candidate,
        const MatchQuality& quality, const std::string& decision, std::size_t selected);
private:
    [[nodiscard]] cv::Mat read(const FrameFeatures& frame) const;
    std::filesystem::path directory_;
    std::vector<std::string> files_;
    std::string title_{"SceneForge keyframes"};
    bool paused_{true};
    bool outliers_{false};
};
}  // namespace sceneforge::video
