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

#include "sceneforge/video/video_extractor.hpp"
#include <atomic>
#include <cstdint>
#include <optional>

namespace sceneforge::video::detail {

struct DecodeSection final {
    std::int64_t begin{};
    std::optional<std::int64_t> end;
    std::int64_t origin{};
    bool seek{false};
    const std::atomic_bool* cancelled{};
    std::int64_t seek_timestamp{};  // Earlier reference frames; begin still owns output.
};

// Owns parallel orchestration; each worker owns a demuxer, decoder, CUDA context
// and bounded writer pool. Only a validated complete sequence is published.
class ParallelExtractor final {
public:
    explicit ParallelExtractor(ExtractionOptions options);
    [[nodiscard]] std::optional<std::size_t> extract(const VideoExtractor::ProgressCallback& progress) const;
private:
    ExtractionOptions options_;
};

}  // namespace sceneforge::video::detail
