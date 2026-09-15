#include "stereoforge/video/keyframe_selector.hpp"
#include "stereoforge/video/keyframe_debug_view.hpp"
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        std::filesystem::path input, output, config;
        bool debug_view = false;
        for (int index = 1; index < argc; ++index) {
            const std::string key = argv[index];
            if (key == "--debug-view") { debug_view = true; continue; }
            if (++index == argc) throw std::invalid_argument("Missing value for " + key);
            if (key == "--input") input = argv[index];
            else if (key == "--output") output = argv[index];
            else if (key == "--config") config = argv[index];
            else throw std::invalid_argument("Unknown option: " + key);
        }
        if (input.empty() || output.empty() || config.empty())
            throw std::invalid_argument("Required: --input FRAME_DIRECTORY --output JSON --config JSON");
        std::ifstream settings(config);
        settings.exceptions(std::ios::failbit | std::ios::badbit);
        nlohmann::json document;
        settings >> document;
        const std::set<std::string> names{"features", "feature_edge", "min_features", "min_inliers", "descriptor_ratio",
            "ransac_pixels", "min_inlier_ratio", "min_coverage", "retention", "motion", "min_sharpness", "min_interval", "tracking_timeout"};
        if (!document.is_object()) throw std::invalid_argument("Keyframe settings must be an object");
        for (nlohmann::json::const_iterator entry = document.begin(); entry != document.end(); ++entry)
            if (!names.contains(entry.key())) throw std::invalid_argument("Unknown keyframe setting: " + entry.key());
        stereoforge::video::KeyframeOptions options;
        options.features = document.value("features", options.features);
        options.feature_edge = document.value("feature_edge", options.feature_edge);
        options.min_features = document.value("min_features", options.min_features);
        options.min_inliers = document.value("min_inliers", options.min_inliers);
        options.descriptor_ratio = document.value("descriptor_ratio", options.descriptor_ratio);
        options.ransac_pixels = document.value("ransac_pixels", options.ransac_pixels);
        options.min_inlier_ratio = document.value("min_inlier_ratio", options.min_inlier_ratio);
        options.min_coverage = document.value("min_coverage", options.min_coverage);
        options.retention = document.value("retention", options.retention);
        options.motion = document.value("motion", options.motion);
        options.min_sharpness = document.value("min_sharpness", options.min_sharpness);
        options.min_interval = document.value("min_interval", options.min_interval);
        options.tracking_timeout = document.value("tracking_timeout", options.tracking_timeout);
        // Parallelism is owned by the bounded feature pool, not nested OpenCV pools.
        cv::setNumThreads(1);
        stereoforge::video::KeyframeSelector selector(options);
        std::unique_ptr<stereoforge::video::KeyframeDebugView> viewer;
        stereoforge::video::KeyframeSelector::DebugCallback debug;
        if (debug_view) {
            if (!std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY"))
                throw std::runtime_error("Debug window needs a desktop display; start Docker with X11 forwarding");
            viewer = std::make_unique<stereoforge::video::KeyframeDebugView>(input);
            debug = [&viewer](const stereoforge::video::FrameFeatures* reference,
                const stereoforge::video::FrameFeatures& candidate, const stereoforge::video::MatchQuality& quality,
                const std::string& decision, std::size_t selected) {
                return viewer->show(reference, candidate, quality, decision, selected);
            };
        }
        const bool success = selector.run(input, output, [](std::size_t processed, std::size_t total, std::size_t selected) {
            std::cout << nlohmann::json{{"saved", processed}, {"total", total}, {"timestamp_seconds", nullptr},
                {"stage", "ORB + RANSAC | " + std::to_string(selected) + " keyframes"}}.dump() << std::endl;
        }, debug);
        if (debug_view) {
            std::cerr << "Debug session saved. Inspect keyframe_selection.status in " << output << '\n';
            return 0;  // Partial/interrupted diagnostics are valid debug results, never a production cache.
        }
        if (!success) std::cerr << "Keyframe selection is not connected or has fewer than three usable keyframes. Report: " << output << '\n';
        return success ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "Keyframe selection failed: " << error.what() << '\n';
        return 1;
    }
}
