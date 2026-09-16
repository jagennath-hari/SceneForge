#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
}

#include <memory>
#include <stdexcept>
#include <string>

namespace stereoforge::video::detail {

struct FormatDeleter { void operator()(AVFormatContext* p) const noexcept { avformat_close_input(&p); } };
struct CodecDeleter { void operator()(AVCodecContext* p) const noexcept { avcodec_free_context(&p); } };
struct FrameDeleter { void operator()(AVFrame* p) const noexcept { av_frame_free(&p); } };
struct PacketDeleter { void operator()(AVPacket* p) const noexcept { av_packet_free(&p); } };
struct ScaleDeleter { void operator()(SwsContext* p) const noexcept { sws_freeContext(p); } };
struct BufferDeleter { void operator()(AVBufferRef* p) const noexcept { av_buffer_unref(&p); } };
using FormatPtr = std::unique_ptr<AVFormatContext, FormatDeleter>;
using CodecPtr = std::unique_ptr<AVCodecContext, CodecDeleter>;
using FramePtr = std::unique_ptr<AVFrame, FrameDeleter>;
using PacketPtr = std::unique_ptr<AVPacket, PacketDeleter>;
using ScalePtr = std::unique_ptr<SwsContext, ScaleDeleter>;
using BufferPtr = std::unique_ptr<AVBufferRef, BufferDeleter>;

inline void check(int result, const char* operation) {
    if (result < 0) {
        char message[AV_ERROR_MAX_STRING_SIZE]{};
        av_strerror(result, message, sizeof(message));
        throw std::runtime_error(std::string(operation) + ": " + message);
    }
}

template<class Pointer>
[[nodiscard]] Pointer require(Pointer pointer) {
    if (!pointer) throw std::bad_alloc{};
    return pointer;
}

}  // namespace stereoforge::video::detail
