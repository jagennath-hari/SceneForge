#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>

namespace stereoforge::video {

struct ExtractionOptions final {
    std::filesystem::path input;
    std::filesystem::path output;
    double start_seconds{0.0};
    std::optional<double> duration;
    std::optional<std::size_t> count;
};

struct ExtractionProgress final {
    std::size_t saved_frames{};
    double timestamp_seconds{};
    std::optional<std::size_t> estimated_total;
};

// Owns configuration. FFmpeg resources are local RAII objects during extraction.
class VideoExtractor final {
public:
    using ProgressCallback = std::function<void(const ExtractionProgress&)>;
    explicit VideoExtractor(ExtractionOptions options);
    [[nodiscard]] std::size_t extract(const ProgressCallback& progress = {}) const;

private:
    ExtractionOptions options_;
};

}  // namespace stereoforge::video
