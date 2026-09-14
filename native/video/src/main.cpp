#include "stereoforge/video/video_extractor.hpp"
#include <nlohmann/json.hpp>
#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

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
        stereoforge::video::ExtractionOptions options;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") {
                std::cout << "stereoforge-extract-frames --input VIDEO --output EMPTY_DIR "
                             "[--start-seconds N] [--duration N] [--frames N]\n";
                return 0;
            }
            if (++i == argc) throw std::invalid_argument("Missing value for " + key);
            const std::string value = argv[i];
            if (key == "--input") options.input = value;
            else if (key == "--output") options.output = value;
            else if (key == "--start-seconds") options.start_seconds = number(value);
            else if (key == "--duration") options.duration = number(value);
            else if (key == "--frames") {
                std::size_t count{};
                const std::from_chars_result parsed = std::from_chars(value.data(), value.data() + value.size(), count);
                if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) throw std::invalid_argument("Invalid frame count");
                options.count = count;
            } else throw std::invalid_argument("Unknown option: " + key);
        }
        if (options.input.empty() || options.output.empty()) throw std::invalid_argument("--input and --output are required");
        const stereoforge::video::VideoExtractor extractor{std::move(options)};
        const std::size_t count = extractor.extract([](const stereoforge::video::ExtractionProgress& p) {
            nlohmann::json event{{"saved", p.saved_frames}, {"timestamp_seconds", p.timestamp_seconds}};
            event["total"] = p.estimated_total ? nlohmann::json(*p.estimated_total) : nlohmann::json(nullptr);
            std::cout << event.dump() << std::endl;
        });
        return count > 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Frame extraction failed: " << error.what() << '\n';
        return 1;
    }
}
