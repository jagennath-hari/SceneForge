#pragma once

#include "stereoforge/video/video_extractor.hpp"
#include <atomic>
#include <cstdint>
#include <optional>

namespace stereoforge::video::detail {

struct DecodeSection final {
    std::int64_t begin{};
    std::optional<std::int64_t> end;
    std::int64_t origin{};
    bool seek{false};
    const std::atomic_bool* cancelled{};
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

}  // namespace stereoforge::video::detail
