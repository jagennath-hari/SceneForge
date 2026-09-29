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

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <cstdint>
#include <map>

struct AVFrame;

namespace sceneforge::video {
namespace detail { struct DecodeSection; class ParallelExtractor; }

struct ExtractionOptions final {
    std::filesystem::path input;
    std::filesystem::path output;
    double start_seconds{0.0};
    std::optional<double> duration;
    std::optional<std::size_t> count;
    int max_edge{1536};  // Geometry working images; zero preserves source resolution.
    bool hardware{true};
    // Exact display timestamps from a prior streaming pass, used for repair.
    std::map<std::int64_t, std::size_t> requested_frames;
    std::optional<double> timestamp_origin;

};

struct ExtractionProgress final {
    std::size_t saved_frames{};
    double timestamp_seconds{};
    std::optional<std::size_t> estimated_total;
    std::string stage{"saving PNGs"};
};

// Owns configuration. FFmpeg resources are local RAII objects during extraction.
class VideoExtractor final {
public:
    using ProgressCallback = std::function<void(const ExtractionProgress&)>;
    using FrameCallback = std::function<bool(const AVFrame&, std::size_t, double, std::int64_t, std::size_t)>;
    // Callback consumes ordered working frames. Only timestamp metadata is saved.
    [[nodiscard]] std::size_t stream(const FrameCallback& frame, const ProgressCallback& progress = {}) const;
    explicit VideoExtractor(ExtractionOptions options);
    [[nodiscard]] std::size_t extract(const ProgressCallback& progress = {}) const;

private:
    friend class detail::ParallelExtractor;
    [[nodiscard]] std::size_t extract_section(const ProgressCallback& progress, int device,
        const detail::DecodeSection* section, unsigned writers) const;
    [[nodiscard]] std::size_t extract_once(const ProgressCallback& progress, bool hardware,
        int device = 0, const detail::DecodeSection* section = nullptr, unsigned writers = 0, const FrameCallback& callback = {}) const;
    ExtractionOptions options_;
};

}  // namespace sceneforge::video
