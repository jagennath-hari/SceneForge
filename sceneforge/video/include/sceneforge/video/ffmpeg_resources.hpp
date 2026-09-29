// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Jagennath Hari
//
// This file is part of SceneForge.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

namespace sceneforge::video::detail {

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

}  // namespace sceneforge::video::detail
