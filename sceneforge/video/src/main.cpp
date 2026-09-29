#include "sceneforge/video/video_extractor.hpp"
#include <nlohmann/json.hpp>
#include <charconv>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <fstream>

namespace {
[[nodiscard]] double number(const std::string& value) {
    std::size_t consumed = 0;
    const double result = std::stod(value, &consumed);
    if (consumed != value.size()) throw std::invalid_argument("Invalid number: " + value);
    return result;
}
}

int main(int argc, char** argv) {
    try {
        sceneforge::video::ExtractionOptions options;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") {
                std::cout << "sceneforge-extract-frames --input VIDEO --output EMPTY_DIR "
                             "[--start-seconds N] [--duration N] [--frames N] "
                             "[--max-edge N (default 1536; 0 = original)] [--hardware cuda|cpu]\n";
                return 0;
            }
            if (++i == argc) throw std::invalid_argument("Missing value for " + key);
            const std::string value = argv[i];
            if (key == "--input") options.input = value;
            else if (key == "--output") options.output = value;
            else if (key == "--recover-manifest") {
                std::ifstream request(value);
                request.exceptions(std::ios::failbit | std::ios::badbit);
                nlohmann::json document; request >> document;
                options.timestamp_origin = document.at("timestamp_origin").get<double>();
                for (const nlohmann::json& frame : document.at("frames")) {
                    if (!options.requested_frames.emplace(frame.at("timestamp_ticks").get<std::int64_t>(),
                        frame.at("source_frame_index").get<std::size_t>()).second)
                        throw std::invalid_argument("Duplicate repair timestamp");
                }
                if (options.requested_frames.empty()) throw std::invalid_argument("Empty repair request");
            }
            else if (key == "--start-seconds") options.start_seconds = number(value);
            else if (key == "--duration") options.duration = number(value);
            else if (key == "--hardware") {
                if (value != "cuda" && value != "cpu") throw std::invalid_argument("Hardware must be cuda or cpu");
                options.hardware = value == "cuda";
            }
            else if (key == "--frames" || key == "--max-edge") {
                std::size_t count{};
                const std::from_chars_result parsed = std::from_chars(value.data(), value.data() + value.size(), count);
                if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) throw std::invalid_argument("Invalid frame count");
                if (key == "--frames") options.count = count;
                else {
                    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                        throw std::invalid_argument("Maximum edge is too large");
                    options.max_edge = static_cast<int>(count);
                }
            } else throw std::invalid_argument("Unknown option: " + key);
        }
        if (options.input.empty() || options.output.empty()) throw std::invalid_argument("--input and --output are required");
        const sceneforge::video::VideoExtractor extractor{std::move(options)};
        const std::size_t count = extractor.extract([](const sceneforge::video::ExtractionProgress& p) {
            nlohmann::json event{{"saved", p.saved_frames}, {"timestamp_seconds", p.timestamp_seconds}};
            event["stage"] = p.stage;
            event["total"] = p.estimated_total ? nlohmann::json(*p.estimated_total) : nlohmann::json(nullptr);
            std::cout << event.dump() << std::endl;
        });
        return count > 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Frame extraction failed: " << error.what() << '\n';
        return 1;
    }
}
