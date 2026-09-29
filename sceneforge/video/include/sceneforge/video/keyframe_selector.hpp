#pragma once

#include <opencv2/core.hpp>
#include "sceneforge/video/video_extractor.hpp"
#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>
#include <string>

namespace sceneforge::video {

class DeviceFeatures;
class RaCoALIKEDExtractor;
class LightGlueMatcher;
inline constexpr char keyframe_frontend[] = "raco_aliked_lightglue";
inline constexpr char keyframe_label[] = "RaCo + LightGlue+ + CUDA RANSAC";

struct FrameFeatures final {
    std::size_t index{};
    double timestamp{};
    cv::Size size;
    double sharpness{};
    std::size_t feature_count{};
    // Streaming keeps pixels only while queued, anchored, or eligible as a bridge.
    cv::Mat image;
    std::int64_t timestamp_ticks{};
    std::size_t source_frame_index{};
    std::vector<cv::KeyPoint> points;
    std::shared_ptr<DeviceFeatures> device_features;
};

struct MatchQuality final {
    std::size_t matches{};
    std::size_t inliers{};
    double ratio{};
    double coverage{};
    double displacement{};  // Median pixel displacement / image diagonal; not 3D parallax.
    bool homography{false};
    std::vector<cv::DMatch> correspondences;
    std::vector<char> inlier_mask;
};

struct KeyframeOptions final {
    int features{1024};
    int min_features{120};
    int min_inliers{60};
    double ransac_pixels{2.0};
    double min_inlier_ratio{0.4};
    double min_coverage{0.25};
    double retention{0.65};
    double motion{0.035};
    double min_sharpness{20.0};
    double min_interval{0.20};
    double tracking_timeout{2.0};
    unsigned workers{1};  // One bounded extraction worker per usable GPU.
    void validate() const;
};

using ExtractorFactory = std::function<std::unique_ptr<RaCoALIKEDExtractor>()>;

class KeyframeSelector final {
public:
    using AcceptanceCallback = std::function<void(std::size_t candidate, std::size_t keyframe, double timestamp)>;
    using ProgressCallback = std::function<void(std::size_t, std::size_t, std::size_t)>;
    // Called in timestamp order on the main thread; false requests a clean stop.
    using DebugCallback = std::function<bool(const FrameFeatures* reference,
        const FrameFeatures& candidate, const MatchQuality&, const std::string& decision,
        std::size_t selected)>;
    explicit KeyframeSelector(KeyframeOptions options, ExtractorFactory factory,
                              std::unique_ptr<LightGlueMatcher> matcher);
    ~KeyframeSelector();
    // Writes a result even for tracking breaks, so diagnostics survive failure.
    [[nodiscard]] bool run(const std::filesystem::path& directory, const std::filesystem::path& output,
                           const ProgressCallback& progress = {}, const DebugCallback& debug = {},
                           const AcceptanceCallback& accepted = {});
    [[nodiscard]] bool run_video(const ExtractionOptions& extraction, const std::filesystem::path& output,
                                 const ProgressCallback& progress = {}, const AcceptanceCallback& accepted = {});
private:
    [[nodiscard]] bool run_impl(const std::filesystem::path& directory, const std::filesystem::path& output,
        const ProgressCallback& progress, const DebugCallback& debug, const AcceptanceCallback& accepted,
        const ExtractionOptions* extraction);
    KeyframeOptions options_;
    ExtractorFactory factory_;
    std::unique_ptr<LightGlueMatcher> matcher_;
};
}  // namespace sceneforge::video
