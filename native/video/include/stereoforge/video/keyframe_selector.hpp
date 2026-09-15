#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>
#include <string>

namespace stereoforge::video {

struct FrameFeatures final {
    std::size_t index{};
    double timestamp{};
    cv::Size size;
    double sharpness{};
    std::vector<cv::KeyPoint> points;
    cv::Mat descriptors;
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
    int features{2000};
    int feature_edge{960};
    int min_features{120};
    int min_inliers{40};
    double descriptor_ratio{0.75};
    double ransac_pixels{2.0};
    double min_inlier_ratio{0.30};
    double min_coverage{0.25};
    double retention{0.65};
    double motion{0.035};
    double min_sharpness{20.0};
    double min_interval{0.20};
    double tracking_timeout{2.0};
    void validate() const;
};

// A learned frontend can implement these interfaces without changing selection.
class FeatureExtractor {
public:
    virtual ~FeatureExtractor() = default;
    [[nodiscard]] virtual FrameFeatures extract(const std::filesystem::path& path,
        std::size_t index, double timestamp) = 0;
};
class FeatureMatcher {
public:
    virtual ~FeatureMatcher() = default;
    [[nodiscard]] virtual MatchQuality match(const FrameFeatures& reference, const FrameFeatures& candidate) = 0;
};
using ExtractorFactory = std::function<std::unique_ptr<FeatureExtractor>()>;

class OrbFeatureExtractor final : public FeatureExtractor {
public:
    explicit OrbFeatureExtractor(const KeyframeOptions& options);
    [[nodiscard]] FrameFeatures extract(const std::filesystem::path& path,
        std::size_t index, double timestamp) override;
private:
    int edge_;
    cv::Ptr<cv::ORB> orb_;
};
class OrbRansacMatcher final : public FeatureMatcher {
public:
    explicit OrbRansacMatcher(KeyframeOptions options);
    [[nodiscard]] MatchQuality match(const FrameFeatures& reference, const FrameFeatures& candidate) override;
private:
    KeyframeOptions options_;
    cv::BFMatcher matcher_{cv::NORM_HAMMING, false};
};

class KeyframeSelector final {
public:
    using ProgressCallback = std::function<void(std::size_t, std::size_t, std::size_t)>;
    // Called in timestamp order on the main thread; false requests a clean stop.
    using DebugCallback = std::function<bool(const FrameFeatures* reference,
        const FrameFeatures& candidate, const MatchQuality&, const std::string& decision,
        std::size_t selected)>;
    explicit KeyframeSelector(KeyframeOptions options, ExtractorFactory factory = {},
                              std::unique_ptr<FeatureMatcher> matcher = {});
    // Writes a result even for tracking breaks, so diagnostics survive failure.
    [[nodiscard]] bool run(const std::filesystem::path& directory, const std::filesystem::path& output,
                           const ProgressCallback& progress = {}, const DebugCallback& debug = {});
private:
    KeyframeOptions options_;
    ExtractorFactory factory_;
    std::unique_ptr<FeatureMatcher> matcher_;
};
}  // namespace stereoforge::video
