#include "stereoforge/video/video_extractor.hpp"
#include "stereoforge/video/ffmpeg_resources.hpp"
#include "stereoforge/video/frame_writer.hpp"
#include "stereoforge/video/cuda_resize.hpp"
#include "stereoforge/video/parallel_extractor.hpp"
extern "C" {
#include <libavutil/pixdesc.h>
}
#include <iostream>

#include <nlohmann/json.hpp>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

namespace stereoforge::video {
namespace {
using namespace detail;

class HardwareDecodeError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

[[nodiscard]] AVPixelFormat choose_format(AVCodecContext*, const AVPixelFormat* formats) {
    for (const AVPixelFormat* format = formats; *format != AV_PIX_FMT_NONE; ++format)
        if (*format == AV_PIX_FMT_CUDA) return *format;
    for (const AVPixelFormat* format = formats; *format != AV_PIX_FMT_NONE; ++format) {
        const AVPixFmtDescriptor* descriptor = av_pix_fmt_desc_get(*format);
        if (descriptor && !(descriptor->flags & AV_PIX_FMT_FLAG_HWACCEL)) return *format;
    }
    return AV_PIX_FMT_NONE;
}

void decode_check(int result, bool hardware, const char* operation) {
    try { check(result, operation); }
    catch (const std::runtime_error& error) {
        if (hardware) throw HardwareDecodeError(error.what());
        throw;
    }
}
}  // namespace

VideoExtractor::VideoExtractor(ExtractionOptions options) : options_(std::move(options)) {
    if (!std::isfinite(this->options_.start_seconds) || this->options_.start_seconds < 0 ||
        (this->options_.duration && (!std::isfinite(*this->options_.duration) || *this->options_.duration <= 0 ||
                              !std::isfinite(this->options_.start_seconds + *this->options_.duration))) ||
        (this->options_.count && *this->options_.count == 0) || this->options_.max_edge < 0 ||
        (this->options_.max_edge > 0 && this->options_.max_edge < 2))
        throw std::invalid_argument("Invalid start, duration, frame count or maximum edge");
}

std::size_t VideoExtractor::extract(const ProgressCallback& progress) const {
    using namespace detail;
    if (!std::filesystem::is_regular_file(this->options_.input)) throw std::runtime_error("Input video does not exist");
    std::filesystem::create_directories(this->options_.output);
    if (!std::filesystem::is_empty(this->options_.output)) throw std::runtime_error("Output directory must be empty");
    if (this->options_.hardware && this->options_.start_seconds == 0 &&
        !this->options_.duration && !this->options_.count) {
        const ParallelExtractor parallel(this->options_);
        const std::optional<std::size_t> count = parallel.extract(progress);
        if (count) return *count;
    }
    return this->extract_section(progress, 0, nullptr, 0);
}

std::size_t VideoExtractor::extract_section(const ProgressCallback& progress, int device,
    const detail::DecodeSection* section, unsigned writers) const {
    try { return this->extract_once(progress, this->options_.hardware, device, section, writers); }
    catch (const HardwareDecodeError& error) {
        std::cerr << "NVIDIA decode unavailable; restarting extraction on CPU: " << error.what() << '\n';
        // This directory was empty on entry and belongs to this extraction.
        for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(this->options_.output))
            std::filesystem::remove(entry.path());
        return this->extract_once(progress, false, device, section, writers);
    }
}

std::size_t VideoExtractor::extract_once(const ProgressCallback& progress, bool hardware,
    int device_index, const detail::DecodeSection* section, unsigned worker_limit) const {
    using namespace detail;
    AVFormatContext* raw = nullptr;
    const int opened = avformat_open_input(&raw, this->options_.input.c_str(), nullptr, nullptr);
    FormatPtr input{raw};
    check(opened, "Open video");
    check(avformat_find_stream_info(input.get(), nullptr), "Read stream metadata");
    int stream_index = -1;
    for (unsigned i = 0; i < input->nb_streams; ++i)
        if (input->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) { stream_index = static_cast<int>(i); break; }
    if (stream_index < 0) throw std::runtime_error("Input has no video stream");
    const AVStream* stream = input->streams[stream_index];
    if (stream->time_base.num <= 0 || stream->time_base.den <= 0) throw std::runtime_error("Invalid video time base");
    const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!codec) throw std::runtime_error("No decoder for this video codec");
    CodecPtr decoder = require(CodecPtr{avcodec_alloc_context3(codec)});
    check(avcodec_parameters_to_context(decoder.get(), stream->codecpar), "Configure decoder");
    decoder->thread_count = 0;  // FFmpeg chooses decoder threads for this machine.
    decoder->err_recognition = AV_EF_CRCCHECK | AV_EF_EXPLODE;
    decoder->pkt_timebase = stream->time_base;
    bool hardware_enabled = false;
    if (hardware) {
        for (int index = 0; ; ++index) {
            const AVCodecHWConfig* configuration = avcodec_get_hw_config(codec, index);
            if (!configuration) break;
            if (configuration->device_type == AV_HWDEVICE_TYPE_CUDA &&
                (configuration->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)) {
                AVBufferRef* device = nullptr;
                const std::string device_name = std::to_string(device_index);
                const int result = av_hwdevice_ctx_create(&device, AV_HWDEVICE_TYPE_CUDA, device_name.c_str(), nullptr, 0);
                if (result >= 0) {
                    decoder->hw_device_ctx = device;
                    decoder->get_format = choose_format;
                    decoder->thread_count = 1;  // NVDEC owns its decode surface queue.
                    hardware_enabled = true;
                } else {
                    av_buffer_unref(&device);
                    std::cerr << "NVIDIA decoder unavailable; using CPU decoding\n";
                }
                break;
            }
        }
    }
    decode_check(avcodec_open2(decoder.get(), codec, nullptr), hardware_enabled, "Open decoder");
    if (section && section->seek) {
        check(av_seek_frame(input.get(), stream_index, section->begin, AVSEEK_FLAG_BACKWARD), "Seek section keyframe");
        avcodec_flush_buffers(decoder.get());
    }
    PacketPtr packet = require(PacketPtr{av_packet_alloc()});
    FramePtr frame = require(FramePtr{av_frame_alloc()});
    FrameWriterPool writers(this->options_.max_edge, worker_limit);
    CudaResizer resizer;
    std::deque<std::future<void>> pending;
    std::size_t saved = 0;
    int source_width = 0;
    int source_height = 0;
    bool used_cuda = false;
    std::optional<double> origin;
    if (stream->start_time != AV_NOPTS_VALUE) origin = stream->start_time * av_q2d(stream->time_base);
    if (section) origin = section->origin * av_q2d(stream->time_base);
    std::optional<double> previous;
    std::optional<std::size_t> total = this->options_.count;
    if (!section && !total && this->options_.start_seconds == 0 && !this->options_.duration && stream->nb_frames > 0)
        total = static_cast<std::size_t>(stream->nb_frames);
    if (progress) progress({0, 0, total});
    nlohmann::json records = nlohmann::json::array();
    const std::function<void()> complete_one = [&] {
        pending.front().get();
        pending.pop_front();
        ++saved;
        records[saved - 1]["bytes"] = std::filesystem::file_size(
            this->options_.output / records[saved - 1]["file"].get<std::string>());
        if (progress) progress({saved, records[saved - 1]["timestamp_seconds"].get<double>(), total});
    };
    bool finished = false;
    // Decode from the beginning even for a diagnostic selection: stable timestamp
    // origin and no seek/keyframe ambiguity. Full-video extraction is the default.
    const std::function<void()> receive = [&, this] {
        while (!finished) {
            if (section && section->cancelled->load()) throw std::runtime_error("Parallel extraction cancelled");
            const int result = avcodec_receive_frame(decoder.get(), frame.get());
            if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) break;
            decode_check(result, hardware_enabled, "Decode frame");
            if (frame->flags & AV_FRAME_FLAG_CORRUPT) throw std::runtime_error("Corrupt video frame");
            if (frame->best_effort_timestamp == AV_NOPTS_VALUE) throw std::runtime_error("Frame has no timestamp");
            const double absolute = frame->best_effort_timestamp * av_q2d(stream->time_base);
            if (!origin) origin = absolute;
            const double timestamp = absolute - *origin;
            if (!std::isfinite(timestamp) || (previous && timestamp <= *previous))
                throw std::runtime_error("Non-increasing or invalid video timestamps");
            previous = timestamp;
            if (section && section->end && frame->best_effort_timestamp >= *section->end) { finished = true; break; }
            if (this->options_.duration && timestamp >= this->options_.start_seconds + *this->options_.duration) { finished = true; break; }
            bool selected = this->options_.start_seconds == 0 || timestamp >= this->options_.start_seconds;
            if (section) selected = frame->best_effort_timestamp >= section->begin;
            if (this->options_.count && this->options_.duration)
                selected = selected && timestamp >= this->options_.start_seconds + records.size() * *this->options_.duration / *this->options_.count;
            if (selected) {
                if (source_width == 0) { source_width = frame->width; source_height = frame->height; }
                if (frame->width != source_width || frame->height != source_height)
                    throw std::runtime_error("Video changes dimensions within the recording");
                if (pending.size() >= writers.capacity()) complete_one();
                FramePtr output;
                if (frame->format == AV_PIX_FMT_CUDA) {
                    const FrameSize size = working_size(frame->width, frame->height, this->options_.max_edge);
                    try { output = resizer.download(*frame, size.width, size.height); }
                    catch (const std::runtime_error& error) { throw HardwareDecodeError(error.what()); }
                    used_cuda = true;
                } else output = require(FramePtr{av_frame_clone(frame.get())});
                std::ostringstream name;
                name << "frame_" << std::setfill('0') << std::setw(6) << records.size() << ".png";
                pending.push_back(writers.submit(std::move(output), this->options_.output / name.str(), records.size()));
                records.push_back({{"file", name.str()}, {"timestamp_seconds", timestamp},
                                   {"timestamp_ticks", frame->best_effort_timestamp}});
                while (!pending.empty() && pending.front().wait_for(std::chrono::seconds(0)) == std::future_status::ready)
                    complete_one();
                if (this->options_.count && records.size() == *this->options_.count) finished = true;
            }
            av_frame_unref(frame.get());
        }
    };
    while (!finished) {
        const int result = av_read_frame(input.get(), packet.get());
        if (result == AVERROR_EOF) {
            decode_check(avcodec_send_packet(decoder.get(), nullptr), hardware_enabled, "Flush decoder");
            receive();
            break;
        }
        check(result, "Read video packet");
        if (packet->stream_index == stream_index) {
            decode_check(avcodec_send_packet(decoder.get(), packet.get()), hardware_enabled, "Send video packet");
            receive();
        }
        av_packet_unref(packet.get());
    }
    if (records.empty() || (this->options_.count && records.size() != *this->options_.count))
        throw std::runtime_error("Video does not contain the requested frames");
    while (!pending.empty()) complete_one();
    const FrameSize size = working_size(source_width, source_height, this->options_.max_edge);
    const std::filesystem::path temporary = this->options_.output / "manifest.json.tmp";
    std::ofstream manifest(temporary);
    manifest.exceptions(std::ios::badbit | std::ios::failbit);
    manifest << nlohmann::json{{"format_version", 1}, {"frames", records},
        {"source_width", source_width}, {"source_height", source_height},
        {"width", size.width}, {"height", size.height}, {"decoder", used_cuda ? "cuda" : "cpu"},
        {"max_edge", this->options_.max_edge}}.dump(2) << '\n';
    manifest.close();
    std::filesystem::rename(temporary, this->options_.output / "manifest.json");
    return records.size();
}
}  // namespace stereoforge::video
