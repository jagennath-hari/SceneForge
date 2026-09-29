// Diagnostic export of existing TensorRT matches; no keyframe selection or GUI.
#include "stereoforge/video/learned_features.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {
[[nodiscard]] nlohmann::json readJson(const std::filesystem::path& path) {
    std::ifstream stream(path);
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    nlohmann::json result;
    stream >> result;
    return result;
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--capabilities") {
            std::cout << "{\"stable_feature_ids\":true,\"device_argument\":true}\n";
            return 0;
        }
        if (argc != 5 && argc != 6) throw std::invalid_argument("Usage: stereoforge-match-pairs REQUEST.json CONFIG.json MODEL_DIR OUTPUT.jsonl [DEVICE]");
        const int device = argc == 6 ? std::stoi(argv[5]) : 0;
        if (device < 0) throw std::invalid_argument("Device must be nonnegative");
        const nlohmann::json request = readJson(argv[1]);
        const nlohmann::json config = readJson(argv[2]);
        stereoforge::video::LearnedOptions learned;
        learned.models = argv[3];
        learned.debug = true; // Copy match coordinates for diagnostic reprojection only.
        learned.width = config.value("model_width", 960);
        learned.height = config.value("model_height", 544);
        learned.features = config.value("features", 1024);
        learned.detector_threshold = config.value("detector_threshold", 0.0f);
        learned.match_threshold = config.value("match_threshold", 0.1f);
        if (config.value("frontend", std::string(stereoforge::video::keyframe_frontend)) != stereoforge::video::keyframe_frontend ||
            learned.width < 64 || learned.height < 64 || learned.width > 4096 || learned.height > 4096 ||
            learned.width % 32 || learned.height % 32 || learned.features < 32 || learned.features > 4096 ||
            !std::isfinite(learned.detector_threshold) || learned.detector_threshold < 0 || learned.detector_threshold >= 1 ||
            !std::isfinite(learned.match_threshold) || learned.match_threshold <= 0 || learned.match_threshold >= 1)
            throw std::invalid_argument("Invalid RaCo/LightGlue configuration");
        stereoforge::video::KeyframeOptions selection;
        selection.features = learned.features;
        selection.ransac_pixels = config.value("ransac_pixels", 2.0);
        selection.validate();
        if (std::filesystem::exists(argv[4])) throw std::invalid_argument("Output already exists");
        std::ofstream output(argv[4]);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        cv::setNumThreads(1);
        stereoforge::video::RaCoALIKEDExtractor extractor(learned, device);
        stereoforge::video::LightGlueMatcher matcher(learned, selection, device);
        std::map<std::size_t, stereoforge::video::FrameFeatures> cache;
        std::map<std::size_t, std::size_t> last_used;
        std::size_t access = 0;
        for (const nlohmann::json& pair : request.at("pairs")) {
            const std::size_t source_index = pair.at("source").get<std::size_t>();
            const std::size_t target_index = pair.at("target").get<std::size_t>();
            const int width = pair.at("width").get<int>();
            const int height = pair.at("height").get<int>();
            if (width <= 0 || height <= 0) throw std::invalid_argument("Invalid image size");
            if (!cache.contains(source_index)) cache.emplace(source_index, extractor.extract(pair.at("source_path").get<std::string>(), source_index, 0));
            if (!cache.contains(target_index)) cache.emplace(target_index, extractor.extract(pair.at("target_path").get<std::string>(), target_index, 0));
            last_used[source_index] = ++access;
            last_used[target_index] = ++access;
            const stereoforge::video::FrameFeatures& a = cache.at(source_index);
            const stereoforge::video::FrameFeatures& b = cache.at(target_index);
            const stereoforge::video::MatchQuality quality = matcher.match(a, b);
            nlohmann::json matches = nlohmann::json::array();
            for (std::size_t i = 0; i < quality.correspondences.size(); ++i) {
                const cv::DMatch& match = quality.correspondences[i];
                const cv::Point2f p = a.points.at(static_cast<std::size_t>(match.queryIdx)).pt;
                const cv::Point2f q = b.points.at(static_cast<std::size_t>(match.trainIdx)).pt;
                // Invert the CUDA resize's half-pixel convention; padding is right/bottom.
                matches.push_back({{"a", {(p.x + 0.5) * width / a.size.width - 0.5,
                                           (p.y + 0.5) * height / a.size.height - 0.5}},
                                   {"b", {(q.x + 0.5) * width / b.size.width - 0.5,
                                           (q.y + 0.5) * height / b.size.height - 0.5}},
                                   {"source_feature", match.queryIdx}, {"target_feature", match.trainIdx},
                                   {"inlier", quality.inlier_mask.at(i) != 0},
                                   {"confidence", 1.0f - match.distance}});
            }
            output << nlohmann::json{{"source", source_index}, {"target", target_index},
                {"verification", quality.homography ? "homography" : "fundamental"},
                {"matches", matches}}.dump() << '\n';
            output.flush();
            std::cout << "Matched " << source_index << " -> " << target_index << ": "
                      << quality.matches << " matches, " << quality.inliers << " RANSAC inliers\n";
            // Distant loop targets break frame-order eviction. Keep recently used
            // source/target features so all pairs for one source reuse extraction.
            while (cache.size() > 32) {
                const std::map<std::size_t, std::size_t>::const_iterator oldest = std::min_element(
                    last_used.cbegin(), last_used.cend(),
                    [](const std::pair<const std::size_t, std::size_t>& left,
                       const std::pair<const std::size_t, std::size_t>& right) { return left.second < right.second; });
                cache.erase(oldest->first);
                last_used.erase(oldest);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
