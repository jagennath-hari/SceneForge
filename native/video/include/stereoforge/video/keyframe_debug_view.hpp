#pragma once

#include "stereoforge/video/keyframe_selector.hpp"
#include <string>
#include <vector>

namespace stereoforge::video {
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
    std::string title_{"StereoForge keyframes"};
    bool paused_{true};
    bool outliers_{false};
};
}  // namespace stereoforge::video
