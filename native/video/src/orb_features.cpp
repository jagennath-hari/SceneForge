#include "stereoforge/video/keyframe_selector.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace stereoforge::video {
void KeyframeOptions::validate() const {
    if (this->features < 120 || this->feature_edge < 64 || this->min_features < 8 ||
        this->min_features > this->features || this->min_inliers < 8 || this->min_inliers > this->features)
        throw std::invalid_argument("Invalid keyframe feature counts or resolution");
    for (double value : {this->descriptor_ratio, this->min_inlier_ratio, this->min_coverage, this->retention, this->motion})
        if (!std::isfinite(value) || value <= 0 || value >= 1) throw std::invalid_argument("Keyframe ratios must be between zero and one");
    for (double value : {this->ransac_pixels, this->tracking_timeout})
        if (!std::isfinite(value) || value <= 0) throw std::invalid_argument("Keyframe thresholds must be positive");
    for (double value : {this->min_sharpness, this->min_interval})
        if (!std::isfinite(value) || value < 0) throw std::invalid_argument("Keyframe thresholds must be nonnegative");
}
OrbFeatureExtractor::OrbFeatureExtractor(const KeyframeOptions& options) :
    edge_(options.feature_edge), orb_(cv::ORB::create(options.features)) {}

FrameFeatures OrbFeatureExtractor::extract(const std::filesystem::path& path, std::size_t index, double timestamp) {
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
    if (image.empty()) throw std::runtime_error("Cannot read candidate image: " + path.string());
    const int edge = std::max(image.cols, image.rows);
    if (edge > this->edge_) {
        const double scale = static_cast<double>(this->edge_) / edge;
        cv::resize(image, image, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    cv::Mat laplacian;
    cv::Laplacian(image, laplacian, CV_32F);
    cv::Scalar mean, deviation;
    cv::meanStdDev(laplacian, mean, deviation);
    FrameFeatures result;
    result.index = index;
    result.timestamp = timestamp;
    result.size = image.size();
    result.sharpness = deviation[0] * deviation[0];
    this->orb_->detectAndCompute(image, cv::noArray(), result.points, result.descriptors);
    result.feature_count = result.points.size();
    return result;
}
OrbRansacMatcher::OrbRansacMatcher(KeyframeOptions options) : options_(std::move(options)) {}

MatchQuality OrbRansacMatcher::match(const FrameFeatures& reference, const FrameFeatures& candidate) {
    MatchQuality result;
    if (reference.descriptors.rows < 2 || candidate.descriptors.rows < 2) return result;
    std::vector<std::vector<cv::DMatch>> forward, reverse;
    this->matcher_.knnMatch(reference.descriptors, candidate.descriptors, forward, 2);
    this->matcher_.knnMatch(candidate.descriptors, reference.descriptors, reverse, 2);
    for (const std::vector<cv::DMatch>& pair : forward) {
        if (pair.size() != 2 || pair[0].distance >= this->options_.descriptor_ratio * pair[1].distance) continue;
        const cv::DMatch& best = pair[0];
        const std::vector<cv::DMatch>& back = reverse.at(best.trainIdx);
        if (back.size() != 2 || back[0].trainIdx != best.queryIdx ||
            back[0].distance >= this->options_.descriptor_ratio * back[1].distance) continue;
        result.correspondences.push_back(best);
    }
    return verifyMatches(reference, candidate, std::move(result.correspondences), this->options_);
}
MatchQuality verifyMatches(const FrameFeatures& reference, const FrameFeatures& candidate,
    std::vector<cv::DMatch> matches, const KeyframeOptions& options) {
    MatchQuality result;
    result.correspondences = std::move(matches);
    std::vector<cv::Point2f> a, b;
    for (const cv::DMatch& match : result.correspondences) {
        a.push_back(reference.points.at(match.queryIdx).pt);
        b.push_back(candidate.points.at(match.trainIdx).pt);
    }
    result.matches = a.size();
    result.inlier_mask.assign(a.size(), 0);
    if (a.size() < 8) return result;
    cv::Mat fundamental_mask, homography_mask;
    const cv::Mat fundamental = cv::findFundamentalMat(a, b, cv::FM_RANSAC,
        options.ransac_pixels, 0.999, 2000, fundamental_mask);
    const cv::Mat homography = cv::findHomography(a, b, cv::RANSAC,
        options.ransac_pixels, homography_mask, 2000, 0.999);
    const int f_count = fundamental.empty() ? 0 : cv::countNonZero(fundamental_mask);
    const int h_count = homography.empty() ? 0 : cv::countNonZero(homography_mask);
    // Homography supports planar facades/pure rotation; neither model proves parallax.
    result.homography = h_count > f_count;
    const cv::Mat& mask = result.homography ? homography_mask : fundamental_mask;
    if (std::max(f_count, h_count) == 0) return result;
    std::array<bool, 16> coverage_a{}, coverage_b{};
    std::vector<double> movement;
    for (std::size_t index = 0; index < a.size(); ++index) {
        if (!mask.ptr<unsigned char>()[index]) continue;
        result.inlier_mask[index] = 1;
        const int ax = std::clamp(static_cast<int>(a[index].x * 4 / reference.size.width), 0, 3);
        const int ay = std::clamp(static_cast<int>(a[index].y * 4 / reference.size.height), 0, 3);
        const int bx = std::clamp(static_cast<int>(b[index].x * 4 / candidate.size.width), 0, 3);
        const int by = std::clamp(static_cast<int>(b[index].y * 4 / candidate.size.height), 0, 3);
        coverage_a[ay * 4 + ax] = true;
        coverage_b[by * 4 + bx] = true;
        movement.push_back(cv::norm(a[index] - b[index]));
    }
    result.inliers = movement.size();
    result.ratio = static_cast<double>(result.inliers) / result.matches;
    result.coverage = static_cast<double>(std::min(std::count(coverage_a.begin(), coverage_a.end(), true),
        std::count(coverage_b.begin(), coverage_b.end(), true))) / 16;
    std::nth_element(movement.begin(), movement.begin() + movement.size() / 2, movement.end());
    result.displacement = movement[movement.size() / 2] / std::hypot(reference.size.width, reference.size.height);
    return result;
}
}  // namespace stereoforge::video
