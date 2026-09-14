#include "stereoforge/video/parallel_extractor.hpp"
#include "stereoforge/video/ffmpeg_resources.hpp"

#include <cuda.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

namespace stereoforge::video::detail {
namespace {

struct PacketIndex final {
    std::vector<std::int64_t> timestamps;
    std::vector<std::int64_t> keyframes;
    std::int64_t origin{};
    std::size_t packet_count{};
    std::size_t discard_count{};
};

// Reading compressed packets is much cheaper than decoding and PNG encoding.
// One packet per displayed frame is only an eligibility condition: the decoded
// frame timestamps must also match this index exactly before publication.
[[nodiscard]] std::optional<PacketIndex> index_video(
    const std::filesystem::path& path, const VideoExtractor::ProgressCallback& progress) {
    AVFormatContext* raw = nullptr;
    const int opened = avformat_open_input(&raw, path.c_str(), nullptr, nullptr);
    FormatPtr input{raw};
    check(opened, "Open video index");
    check(avformat_find_stream_info(input.get(), nullptr), "Read video index metadata");
    int stream_index = -1;
    for (unsigned index = 0; index < input->nb_streams; ++index)
        if (input->streams[index]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            stream_index = static_cast<int>(index); break;
        }
    if (stream_index < 0) return std::nullopt;
    const AVStream* stream = input->streams[stream_index];
    if (stream->time_base.num <= 0 || stream->time_base.den <= 0) return std::nullopt;
    const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!codec) return std::nullopt;
    bool cuda_supported = false;
    for (int index = 0; ; ++index) {
        const AVCodecHWConfig* configuration = avcodec_get_hw_config(codec, index);
        if (!configuration) break;
        if (configuration->device_type == AV_HWDEVICE_TYPE_CUDA &&
            (configuration->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)) cuda_supported = true;
    }
    if (!cuda_supported) return std::nullopt;
    if (progress) progress({0, 0, std::nullopt, "indexing video for multiple GPUs"});
    PacketIndex result;
    PacketPtr packet = require(PacketPtr{av_packet_alloc()});
    while (true) {
        const int status = av_read_frame(input.get(), packet.get());
        if (status == AVERROR_EOF) break;
        check(status, "Index video packet");
        if (packet->stream_index == stream_index) {
            if (packet->pts == AV_NOPTS_VALUE || packet->flags & AV_PKT_FLAG_CORRUPT) return std::nullopt;
            ++result.packet_count;
            if (packet->flags & AV_PKT_FLAG_KEY) result.keyframes.push_back(packet->pts);
            // These packets still go to the decoder to establish references,
            // but FFmpeg explicitly marks them as not belonging to the output.
            if (packet->flags & AV_PKT_FLAG_DISCARD) ++result.discard_count;
            else result.timestamps.push_back(packet->pts);
            if (progress && result.packet_count % 256 == 0)
                progress({0, 0, std::nullopt, "indexing video: " + std::to_string(result.packet_count) + " packets"});
        }
        av_packet_unref(packet.get());
    }
    if (result.timestamps.empty() || result.keyframes.size() < 2) return std::nullopt;
    std::sort(result.timestamps.begin(), result.timestamps.end());
    std::sort(result.keyframes.begin(), result.keyframes.end());
    if (std::adjacent_find(result.timestamps.begin(), result.timestamps.end()) != result.timestamps.end())
        return std::nullopt;
    // Container sample counts can include edit-list preroll. Validate displayed
    // timestamps against the decoder output instead of treating nb_frames as truth.
    result.origin = stream->start_time == AV_NOPTS_VALUE ? result.timestamps.front() : stream->start_time;
    return result;
}

[[nodiscard]] std::vector<std::int64_t> partition(const PacketIndex& index, int devices) {
    std::vector<std::int64_t> boundaries{index.timestamps.front()};
    for (int device = 1; device < devices; ++device) {
        const std::int64_t target = index.timestamps[index.timestamps.size() * device / devices];
        const std::vector<std::int64_t>::const_iterator key =
            std::lower_bound(index.keyframes.begin(), index.keyframes.end(), target);
        if (key != index.keyframes.end() && *key > boundaries.back()) boundaries.push_back(*key);
    }
    return boundaries;
}

[[nodiscard]] nlohmann::json read_manifest(const std::filesystem::path& directory) {
    std::ifstream input(directory / "manifest.json");
    input.exceptions(std::ios::badbit | std::ios::failbit);
    nlohmann::json document;
    input >> document;
    return document;
}

}  // namespace

ParallelExtractor::ParallelExtractor(ExtractionOptions options) : options_(std::move(options)) {}

std::optional<std::size_t> ParallelExtractor::extract(const VideoExtractor::ProgressCallback& progress) const {
    int devices = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&devices) != CUDA_SUCCESS || devices < 2)
        return std::nullopt;
    const std::optional<PacketIndex> indexed = index_video(this->options_.input, progress);
    if (!indexed) return std::nullopt;
    const PacketIndex& index = *indexed;
    const std::vector<std::int64_t> boundaries = partition(index, devices);
    if (boundaries.size() < 2) return std::nullopt;
    const unsigned writer_budget = std::clamp(std::thread::hardware_concurrency(), 1u, 8u);
    const unsigned writers = std::max(1u, writer_budget / static_cast<unsigned>(boundaries.size()));
    const std::filesystem::path workspace = this->options_.output / ".sections";
    std::filesystem::create_directory(workspace);
    std::atomic_bool cancelled{false};
    std::mutex progress_mutex;
    std::vector<std::size_t> completed(boundaries.size(), 0);
    std::vector<double> times(boundaries.size(), 0.0);
    std::vector<std::future<std::size_t>> tasks;
    std::exception_ptr failure;
    nlohmann::json diagnostic{{"packet_count", index.packet_count},
        {"decode_only_packets", index.discard_count}, {"expected_frames", index.timestamps.size()},
        {"sections", nlohmann::json::array()}};
    try {
        for (std::size_t device = 0; device < boundaries.size(); ++device) {
            ExtractionOptions options = this->options_;
            options.output = workspace / std::to_string(device);
            std::filesystem::create_directory(options.output);
            const std::vector<std::int64_t>::const_iterator boundary_key =
                std::lower_bound(index.keyframes.begin(), index.keyframes.end(), boundaries[device]);
            const std::size_t key_index = static_cast<std::size_t>(boundary_key - index.keyframes.begin());
            const std::int64_t preroll = index.keyframes[key_index > 2 ? key_index - 2 : 0];
            const bool seek = device != 0 && preroll > index.timestamps.front();
            const DecodeSection section{boundaries[device],
                device + 1 < boundaries.size() ? std::optional<std::int64_t>(boundaries[device + 1]) : std::nullopt,
                index.origin, seek, &cancelled, preroll};
            diagnostic["sections"].push_back({{"device", device}, {"begin_ticks", section.begin},
                {"end_ticks", section.end ? nlohmann::json(*section.end) : nlohmann::json(nullptr)},
                {"seek_ticks", seek ? nlohmann::json(preroll) : nlohmann::json(nullptr)}});
            tasks.push_back(std::async(std::launch::async, [&, options, section, device] {
                try {
                    const VideoExtractor extractor(options);
                    return extractor.extract_section([&](const ExtractionProgress& event) {
                        const std::lock_guard<std::mutex> lock(progress_mutex);
                        completed[device] = event.saved_frames;
                        times[device] = event.timestamp_seconds;
                        if (progress) progress({std::accumulate(completed.begin(), completed.end(), std::size_t{0}),
                            *std::max_element(times.begin(), times.end()), index.timestamps.size(),
                            "saving PNGs across " + std::to_string(boundaries.size()) + " devices"});
                    }, static_cast<int>(device), &section, writers);
                } catch (...) { cancelled.store(true); throw; }
            }));
        }
    } catch (...) { cancelled.store(true); failure = std::current_exception(); }
    // Join every worker before touching its files, including after a failure.
    for (std::future<std::size_t>& task : tasks) {
        try { static_cast<void>(task.get()); }
        catch (...) { cancelled.store(true); if (!failure) failure = std::current_exception(); }
    }
    try {
        if (failure) std::rethrow_exception(failure);
        nlohmann::json merged;
        nlohmann::json records = nlohmann::json::array();
        nlohmann::json sections = nlohmann::json::array();
        std::vector<nlohmann::json> documents;
        std::size_t frame_index = 0;
        double previous = -std::numeric_limits<double>::infinity();
        bool any_cpu = false;
        bool any_cuda = false;
        for (std::size_t device = 0; device < boundaries.size(); ++device) {
            nlohmann::json document = read_manifest(workspace / std::to_string(device));
            diagnostic["sections"][device]["saved_frames"] = document.at("frames").size();
            diagnostic["sections"][device]["first_ticks"] = document.at("frames").front().at("timestamp_ticks");
            diagnostic["sections"][device]["last_ticks"] = document.at("frames").back().at("timestamp_ticks");
            if (device == 0) merged = document;
            for (const char* field : {"source_width", "source_height", "width", "height", "max_edge"})
                if (document.at(field) != merged.at(field)) throw std::runtime_error("Section image dimensions differ");
            for (const nlohmann::json& frame : document.at("frames")) {
                if (frame_index >= index.timestamps.size() ||
                    frame.at("timestamp_ticks").get<std::int64_t>() != index.timestamps[frame_index]) {
                    const std::string expected = frame_index < index.timestamps.size()
                        ? std::to_string(index.timestamps[frame_index]) : "end of sequence";
                    diagnostic["first_mismatch"] = {{"frame_index", frame_index}, {"device", device},
                        {"expected_ticks", expected}, {"actual_ticks", frame.at("timestamp_ticks")}};
                    throw std::runtime_error("Timestamp mismatch at frame " + std::to_string(frame_index) +
                        ", device " + std::to_string(device) + ": expected " + expected +
                        ", got " + frame.at("timestamp_ticks").dump());
                }
                const double timestamp = frame.at("timestamp_seconds").get<double>();
                if (!std::isfinite(timestamp) || timestamp <= previous)
                    throw std::runtime_error("Section timestamps are not strictly ordered");
                previous = timestamp;
                ++frame_index;
            }
            any_cpu = any_cpu || document.at("decoder") == "cpu";
            any_cuda = any_cuda || document.at("decoder") == "cuda";
            sections.push_back({{"device", device}, {"decoder", document.at("decoder")},
                                {"frames", document.at("frames").size()}});
            documents.push_back(std::move(document));
        }
        if (frame_index != index.timestamps.size())
            throw std::runtime_error("Parallel extraction saved " + std::to_string(frame_index) +
                " of " + std::to_string(index.timestamps.size()) + " expected displayed frames");
        if (progress) progress({frame_index, previous, frame_index, "publishing ordered frames"});
        for (std::size_t device = 0; device < documents.size(); ++device) {
            for (nlohmann::json frame : documents[device].at("frames")) {
                std::ostringstream name;
                name << "frame_" << std::setfill('0') << std::setw(6) << records.size() << ".png";
                std::filesystem::rename(workspace / std::to_string(device) / frame.at("file").get<std::string>(),
                                        this->options_.output / name.str());
                frame["file"] = name.str();
                records.push_back(std::move(frame));
            }
        }
        merged["frames"] = std::move(records);
        merged["decoder"] = any_cpu ? (any_cuda ? "mixed" : "cpu") : "cuda";
        merged["sections"] = std::move(sections);
        merged["decode_only_packets"] = index.discard_count;
        std::ofstream output(this->options_.output / "manifest.json.tmp");
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output << merged.dump(2) << '\n';
        output.close();
        std::filesystem::rename(this->options_.output / "manifest.json.tmp", this->options_.output / "manifest.json");
        std::filesystem::remove_all(workspace);
        return frame_index;
    } catch (const std::exception& error) {
        std::cerr << "Parallel extraction could not be validated; restarting sequentially: " << error.what() << '\n';
        diagnostic["error"] = error.what();
        diagnostic["saved_frames"] = std::accumulate(completed.begin(), completed.end(), std::size_t{0});
        const std::filesystem::path diagnostic_path = this->options_.output / "parallel_diagnostic.json";
        std::ofstream report(diagnostic_path);
        report.exceptions(std::ios::badbit | std::ios::failbit);
        report << diagnostic.dump(2) << '\n';
        report.close();
        std::cerr << "Parallel extraction diagnostic: " << diagnostic_path << '\n';
        // The caller required an empty destination; all contents belong to us.
        for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(this->options_.output))
            if (entry.path() != diagnostic_path) std::filesystem::remove_all(entry.path());
        return std::nullopt;
    }
}
}  // namespace stereoforge::video::detail
