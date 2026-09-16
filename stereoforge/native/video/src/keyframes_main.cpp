#include "stereoforge/video/keyframe_selector.hpp"
#include "stereoforge/video/keyframe_debug_view.hpp"
#include "stereoforge/video/learned_features.hpp"
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        std::filesystem::path input, output, config, models;
        bool debug_view = false;
        for (int index = 1; index < argc; ++index) {
            const std::string key = argv[index];
            if (key == "--debug-view") { debug_view = true; continue; }
            if (++index == argc) throw std::invalid_argument("Missing value for " + key);
            if (key == "--input") input = argv[index];
            else if (key == "--output") output = argv[index];
            else if (key == "--models") models = argv[index];
            else if (key == "--config") config = argv[index];
            else throw std::invalid_argument("Unknown option: " + key);
        }
        if (input.empty() || output.empty() || config.empty())
            throw std::invalid_argument("Required: --input FRAME_DIRECTORY --output JSON --config JSON");
        std::ifstream settings(config);
        settings.exceptions(std::ios::failbit | std::ios::badbit);
        nlohmann::json document;
        settings >> document;
        const std::set<std::string> names{"features", "min_features", "min_inliers",
            "ransac_pixels", "min_inlier_ratio", "min_coverage", "retention", "motion", "min_sharpness", "min_interval", "tracking_timeout", "frontend", "detector_threshold", "match_threshold", "model_width", "model_height", "precision"};
        if (!document.is_object()) throw std::invalid_argument("Keyframe settings must be an object");
        for (nlohmann::json::const_iterator entry = document.begin(); entry != document.end(); ++entry)
            if (!names.contains(entry.key())) throw std::invalid_argument("Unknown keyframe setting: " + entry.key());
        stereoforge::video::KeyframeOptions options;
        if (document.value("frontend", std::string(stereoforge::video::keyframe_frontend)) != stereoforge::video::keyframe_frontend)
            throw std::invalid_argument("Only raco_aliked_lightglue keyframes are supported");
        options.features = document.value("features", options.features);
        options.min_features = document.value("min_features", options.min_features);
        options.min_inliers = document.value("min_inliers", options.min_inliers);
        options.ransac_pixels = document.value("ransac_pixels", options.ransac_pixels);
        options.min_inlier_ratio = document.value("min_inlier_ratio", options.min_inlier_ratio);
        options.min_coverage = document.value("min_coverage", options.min_coverage);
        options.retention = document.value("retention", options.retention);
        options.motion = document.value("motion", options.motion);
        options.min_sharpness = document.value("min_sharpness", options.min_sharpness);
        options.min_interval = document.value("min_interval", options.min_interval);
        options.tracking_timeout = document.value("tracking_timeout", options.tracking_timeout);
        options.validate();
        stereoforge::video::ExtractorFactory factory;
        std::unique_ptr<stereoforge::video::LightGlueMatcher> matcher;
        stereoforge::video::LearnedOptions learned;
        learned.models = models;
        learned.debug = debug_view;
        learned.features = options.features;
        learned.width = document.value("model_width", 960);
        learned.height = document.value("model_height", 544);
        learned.detector_threshold = document.value("detector_threshold", 0.0f);
        learned.match_threshold = document.value("match_threshold", 0.1f);
        if (models.empty() || learned.width < 64 || learned.height < 64 ||
            learned.width % 32 || learned.height % 32 || learned.width > 4096 || learned.height > 4096 ||
            learned.features > 4096 || learned.features > learned.width * learned.height ||
            !std::isfinite(learned.detector_threshold) || learned.detector_threshold < 0 || learned.detector_threshold >= 1 ||
            !std::isfinite(learned.match_threshold) || learned.match_threshold <= 0 || learned.match_threshold >= 1)
            throw std::invalid_argument("Invalid learned model settings or missing --models; use the Python demo to prepare engines");
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1)
            throw std::runtime_error("RaCo–ALIKED/LightGlue requires a visible CUDA GPU");
        options.workers = static_cast<unsigned>(devices);
        for (int device = 1; device < devices; ++device) {
            int accessible = 0;
            if (cudaDeviceCanAccessPeer(&accessible, 0, device) != cudaSuccess)
                throw std::runtime_error("Cannot inspect CUDA peer access");
            if (!accessible) {
                options.workers = 1;
                std::cerr << "GPU peer access unavailable; keeping feature extraction and matching on GPU 0\n";
                break;
            }
        }
        const std::shared_ptr<std::atomic<int>> next = std::make_shared<std::atomic<int>>(0);
        factory = [learned, next] {
            return std::make_unique<stereoforge::video::RaCoALIKEDExtractor>(learned, next->fetch_add(1));
        };
        matcher = std::make_unique<stereoforge::video::LightGlueMatcher>(learned, options, 0);
        // Parallelism is owned by the bounded feature pool, not nested OpenCV pools.
        cv::setNumThreads(1);
        stereoforge::video::KeyframeSelector selector(options, std::move(factory), std::move(matcher));
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
        const std::string frontend_label = stereoforge::video::keyframe_label;
        const bool success = selector.run(input, output, [&frontend_label](std::size_t processed, std::size_t total, std::size_t selected) {
            std::cout << nlohmann::json{{"saved", processed}, {"total", total}, {"timestamp_seconds", nullptr},
                {"stage", frontend_label + " | " + std::to_string(selected) + " keyframes"}}.dump() << std::endl;
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
