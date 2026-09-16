#include "stereoforge/video/keyframe_debug_view.hpp"
#include <nlohmann/json.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace stereoforge::video {
KeyframeDebugView::KeyframeDebugView(const std::filesystem::path& directory) : directory_(directory) {
    std::ifstream input(this->directory_ / "manifest.json");
    input.exceptions(std::ios::badbit | std::ios::failbit);
    nlohmann::json manifest;
    input >> manifest;
    for (const nlohmann::json& frame : manifest.at("frames"))
        this->files_.push_back(frame.at("file").get<std::string>());
    this->title_ += std::string(" | ") + keyframe_label;
    cv::namedWindow(this->title_, cv::WINDOW_NORMAL);
    cv::resizeWindow(this->title_, 1440, 700);
}
KeyframeDebugView::~KeyframeDebugView() {
    try { cv::destroyWindow(this->title_); } catch (const cv::Exception&) {}
}
cv::Mat KeyframeDebugView::read(const FrameFeatures& frame) const {
    cv::Mat image = cv::imread((this->directory_ / this->files_.at(frame.index)).string());
    if (image.empty()) throw std::runtime_error("Cannot read debug frame");
    cv::resize(image, image, frame.size, 0, 0, cv::INTER_AREA);
    return image;
}
bool KeyframeDebugView::show(const FrameFeatures* reference, const FrameFeatures& candidate,
    const MatchQuality& quality, const std::string& decision, std::size_t selected) {
    const cv::Mat right = this->read(candidate);
    cv::Mat left;
    if (reference) left = this->read(*reference);
    else left = cv::Mat::zeros(right.size(), right.type());
    while (true) {
        cv::Mat matches;
        if (reference) {
            cv::drawMatches(left, reference->points, right, candidate.points, quality.correspondences,
                matches, cv::Scalar(60, 220, 60), cv::Scalar::all(-1), quality.inlier_mask,
                cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            if (this->outliers_) {
                for (std::size_t index = 0; index < quality.correspondences.size(); ++index) {
                    if (quality.inlier_mask.at(index)) continue;
                    const cv::DMatch& match = quality.correspondences[index];
                    const cv::Point2f a = reference->points.at(match.queryIdx).pt;
                    const cv::Point2f b = candidate.points.at(match.trainIdx).pt + cv::Point2f(left.cols, 0);
                    cv::line(matches, a, b, cv::Scalar(60, 60, 230), 1, cv::LINE_AA);
                }
            }
        } else cv::hconcat(left, right, matches);
        // Scale the image area only; keep captions readable regardless of source resolution.
        cv::Mat scaled;
        const double scale = 1440.0 / matches.cols;
        cv::resize(matches, scaled, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::Mat canvas;
        cv::copyMakeBorder(scaled, canvas, 130, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(25, 25, 25));
        std::ostringstream metrics;
        metrics << std::fixed << std::setprecision(3) << "Matches " << quality.matches
            << " | inliers " << quality.inliers << " | ratio " << quality.ratio
            << " | coverage " << quality.coverage << " | motion " << quality.displacement
            << " | " << (quality.inliers == 0 ? "no verified model" : (quality.homography ? "H" : "F"));
        std::ostringstream candidate_label;
        candidate_label << std::fixed << std::setprecision(2) << "Candidate " << candidate.index
            << " @ " << candidate.timestamp << "s | features " << candidate.feature_count
            << " | sharpness " << candidate.sharpness;
        const std::vector<std::string> lines{
            (reference ? "Reference keyframe " + std::to_string(reference->index) : "No reference yet")
                + "    ->    " + candidate_label.str(),
            "Decision: " + decision + " | accepted keyframes: " + std::to_string(selected),
            metrics.str(),
            std::string(this->paused_ ? "PAUSED" : "PLAYING")
                + " | Space: play/pause | N: next | O: outliers | Q/Esc: quit | green: RANSAC inliers"};
        for (std::size_t index = 0; index < lines.size(); ++index)
            cv::putText(canvas, lines[index], cv::Point(12, 25 + 29 * static_cast<int>(index)),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
        cv::imshow(this->title_, canvas);
        // Poll even when paused so closing the window terminates cleanly.
        const int key = cv::waitKey(30) & 0xff;
        if (key == 27 || key == 'q' || cv::getWindowProperty(this->title_, cv::WND_PROP_VISIBLE) < 1) return false;
        if (key == ' ') this->paused_ = !this->paused_;
        if (key == 'o') { this->outliers_ = !this->outliers_; continue; }
        if (key == 'n') { this->paused_ = true; return true; }
        if (!this->paused_) return true;
    }
}
}  // namespace stereoforge::video
