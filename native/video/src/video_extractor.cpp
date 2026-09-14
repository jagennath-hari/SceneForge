#include "stereoforge/video/video_extractor.hpp"
#include "ffmpeg_resources.hpp"

#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

namespace stereoforge::video {
namespace {
using namespace detail;

class PngWriter final {
public:
    explicit PngWriter(const AVFrame& source) :
        rgb_(require(FramePtr{av_frame_alloc()})), packet_(require(PacketPtr{av_packet_alloc()})) {
        const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_PNG);
        if (!codec) throw std::runtime_error("FFmpeg PNG encoder is unavailable");
        this->encoder_ = require(CodecPtr{avcodec_alloc_context3(codec)});
        this->encoder_->width = source.width;
        this->encoder_->height = source.height;
        this->encoder_->pix_fmt = AV_PIX_FMT_RGB24;
        this->encoder_->time_base = AVRational{1, 1};
        this->encoder_->compression_level = 1;  // Lossless; favor extraction speed over file size.
        check(avcodec_open2(this->encoder_.get(), codec, nullptr), "Open PNG encoder");
        this->rgb_->width = source.width;
        this->rgb_->height = source.height;
        this->rgb_->format = AV_PIX_FMT_RGB24;
        check(av_frame_get_buffer(this->rgb_.get(), 32), "Allocate RGB frame");
    }

    void write(const AVFrame& source, const std::filesystem::path& path, std::size_t index) {
        if (source.width != this->rgb_->width || source.height != this->rgb_->height)
            throw std::runtime_error("Video changes dimensions within the recording");
        check(av_frame_make_writable(this->rgb_.get()), "Make RGB buffer writable");
        // FFmpeg may change pixel format between frames. Recreate as necessary.
        if (!this->scale_ || source.format != this->format_) {
            this->scale_.reset(sws_getContext(source.width, source.height, static_cast<AVPixelFormat>(source.format),
                                       this->rgb_->width, this->rgb_->height, AV_PIX_FMT_RGB24, SWS_BILINEAR,
                                       nullptr, nullptr, nullptr));
            if (!this->scale_) throw std::runtime_error("Cannot create RGB conversion context");
            this->format_ = source.format;
        }
        const int* coefficients = sws_getCoefficients(source.colorspace == AVCOL_SPC_UNSPECIFIED
                                                       ? SWS_CS_DEFAULT : static_cast<int>(source.colorspace));
        check(sws_setColorspaceDetails(this->scale_.get(), coefficients, source.color_range == AVCOL_RANGE_JPEG,
                                      coefficients, 1, 0, 1 << 16, 1 << 16), "Configure color conversion");
        if (sws_scale(this->scale_.get(), source.data, source.linesize, 0, source.height,
                      this->rgb_->data, this->rgb_->linesize) != source.height)
            throw std::runtime_error("Incomplete RGB conversion");
        this->rgb_->pts = static_cast<std::int64_t>(index);
        check(avcodec_send_frame(this->encoder_.get(), this->rgb_.get()), "Encode PNG");
        // PNG is an intra-frame encoder with no delayed frames: one packet per image.
        check(avcodec_receive_packet(this->encoder_.get(), this->packet_.get()), "Receive PNG");
        std::ofstream output(path, std::ios::binary);
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output.write(reinterpret_cast<const char*>(this->packet_->data), this->packet_->size);
        output.close();
        av_packet_unref(this->packet_.get());
    }

private:
    CodecPtr encoder_;
    FramePtr rgb_;
    PacketPtr packet_;
    ScalePtr scale_;
    int format_{-1};
};
}  // namespace

VideoExtractor::VideoExtractor(ExtractionOptions options) : options_(std::move(options)) {
    if (!std::isfinite(this->options_.start_seconds) || this->options_.start_seconds < 0 ||
        (this->options_.duration && (!std::isfinite(*this->options_.duration) || *this->options_.duration <= 0 ||
                              !std::isfinite(this->options_.start_seconds + *this->options_.duration))) ||
        (this->options_.count && *this->options_.count == 0))
        throw std::invalid_argument("Invalid start, duration or frame count");
}

std::size_t VideoExtractor::extract(const ProgressCallback& progress) const {
    using namespace detail;
    if (!std::filesystem::is_regular_file(this->options_.input)) throw std::runtime_error("Input video does not exist");
    std::filesystem::create_directories(this->options_.output);
    if (!std::filesystem::is_empty(this->options_.output)) throw std::runtime_error("Output directory must be empty");
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
    check(avcodec_open2(decoder.get(), codec, nullptr), "Open decoder");
    PacketPtr packet = require(PacketPtr{av_packet_alloc()});
    FramePtr frame = require(FramePtr{av_frame_alloc()});
    std::unique_ptr<PngWriter> writer;
    std::optional<double> origin;
    if (stream->start_time != AV_NOPTS_VALUE) origin = stream->start_time * av_q2d(stream->time_base);
    std::optional<double> previous;
    std::optional<std::size_t> total = this->options_.count;
    if (!total && this->options_.start_seconds == 0 && !this->options_.duration && stream->nb_frames > 0)
        total = static_cast<std::size_t>(stream->nb_frames);
    if (progress) progress({0, 0, total});
    nlohmann::json records = nlohmann::json::array();
    bool finished = false;
    // Decode from the beginning even for a diagnostic selection: stable timestamp
    // origin and no seek/keyframe ambiguity. Full-video extraction is the default.
    const std::function<void()> receive = [&, this] {
        while (!finished) {
            const int result = avcodec_receive_frame(decoder.get(), frame.get());
            if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) break;
            check(result, "Decode frame");
            if (frame->flags & AV_FRAME_FLAG_CORRUPT) throw std::runtime_error("Corrupt video frame");
            if (frame->best_effort_timestamp == AV_NOPTS_VALUE) throw std::runtime_error("Frame has no timestamp");
            const double absolute = frame->best_effort_timestamp * av_q2d(stream->time_base);
            if (!origin) origin = absolute;
            const double timestamp = absolute - *origin;
            if (!std::isfinite(timestamp) || (previous && timestamp <= *previous))
                throw std::runtime_error("Non-increasing or invalid video timestamps");
            previous = timestamp;
            if (this->options_.duration && timestamp >= this->options_.start_seconds + *this->options_.duration) { finished = true; break; }
            bool selected = this->options_.start_seconds == 0 || timestamp >= this->options_.start_seconds;
            if (this->options_.count && this->options_.duration)
                selected = selected && timestamp >= this->options_.start_seconds + records.size() * *this->options_.duration / *this->options_.count;
            if (selected) {
                if (!writer) writer = std::make_unique<PngWriter>(*frame);
                std::ostringstream name;
                name << "frame_" << std::setfill('0') << std::setw(6) << records.size() << ".png";
                writer->write(*frame, this->options_.output / name.str(), records.size());
                records.push_back({{"file", name.str()}, {"timestamp_seconds", timestamp}});
                if (progress) progress({records.size(), timestamp, total});
                if (this->options_.count && records.size() == *this->options_.count) finished = true;
            }
            av_frame_unref(frame.get());
        }
    };
    while (!finished) {
        const int result = av_read_frame(input.get(), packet.get());
        if (result == AVERROR_EOF) {
            check(avcodec_send_packet(decoder.get(), nullptr), "Flush decoder");
            receive();
            break;
        }
        check(result, "Read video packet");
        if (packet->stream_index == stream_index) {
            check(avcodec_send_packet(decoder.get(), packet.get()), "Send video packet");
            receive();
        }
        av_packet_unref(packet.get());
    }
    if (records.empty() || (this->options_.count && records.size() != *this->options_.count))
        throw std::runtime_error("Video does not contain the requested frames");
    const std::filesystem::path temporary = this->options_.output / "manifest.json.tmp";
    std::ofstream manifest(temporary);
    manifest.exceptions(std::ios::badbit | std::ios::failbit);
    manifest << nlohmann::json{{"format_version", 1}, {"frames", records}}.dump(2) << '\n';
    manifest.close();
    std::filesystem::rename(temporary, this->options_.output / "manifest.json");
    return records.size();
}
}  // namespace stereoforge::video
