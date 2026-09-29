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

#include "sceneforge/video/ffmpeg_resources.hpp"
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <fstream>
#include <future>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#include <filesystem>

namespace sceneforge::video::detail {

struct FrameSize final { int width; int height; };

[[nodiscard]] inline FrameSize working_size(int width, int height, int max_edge) {
    if (max_edge == 0 || std::max(width, height) <= max_edge) return {width, height};
    const double factor = static_cast<double>(max_edge) / std::max(width, height);
    // Even dimensions preserve 4:2:0 chroma layout. No crop or upscaling.
    return {std::max(2, static_cast<int>(width * factor) / 2 * 2),
            std::max(2, static_cast<int>(height * factor) / 2 * 2)};
}

// Same color conversion for streaming inference and lossless image export.
class RgbConverter final {
public:
    RgbConverter(const AVFrame& source, int max_edge) : rgb_(require(FramePtr{av_frame_alloc()})) {
        const FrameSize size = working_size(source.width,source.height,max_edge);
        this->rgb_->width = size.width; this->rgb_->height = size.height;
        this->rgb_->format = AV_PIX_FMT_RGB24;
        check(av_frame_get_buffer(this->rgb_.get(),32), "Allocate RGB frame");
    }
    [[nodiscard]] AVFrame& convert(const AVFrame& source) {
        check(av_frame_make_writable(this->rgb_.get()), "Make RGB buffer writable");
        // FFmpeg may change pixel format between frames. Recreate as necessary.
        if (!this->scale_ || source.format != this->format_ || source.width != this->width_ || source.height != this->height_) {
            this->scale_.reset(sws_getContext(source.width, source.height, static_cast<AVPixelFormat>(source.format),
                                       this->rgb_->width, this->rgb_->height, AV_PIX_FMT_RGB24, SWS_AREA,
                                       nullptr, nullptr, nullptr));
            if (!this->scale_) throw std::runtime_error("Cannot create RGB conversion context");
            this->format_ = source.format;
            this->width_ = source.width;
            this->height_ = source.height;
        }
        const int* coefficients = sws_getCoefficients(source.colorspace == AVCOL_SPC_UNSPECIFIED
                                                       ? SWS_CS_DEFAULT : static_cast<int>(source.colorspace));
        check(sws_setColorspaceDetails(this->scale_.get(), coefficients, source.color_range == AVCOL_RANGE_JPEG,
                                      coefficients, 1, 0, 1 << 16, 1 << 16), "Configure color conversion");
        if (sws_scale(this->scale_.get(), source.data, source.linesize, 0, source.height,
                      this->rgb_->data, this->rgb_->linesize) != this->rgb_->height)
            throw std::runtime_error("Incomplete RGB conversion");
        return *this->rgb_;
    }

private:
    FramePtr rgb_;
    ScalePtr scale_;
    int format_{-1};
    int width_{};
    int height_{};
};

class PngWriter final {
public:
    PngWriter(const AVFrame& source, int max_edge) :
        converter_(source,max_edge), packet_(require(PacketPtr{av_packet_alloc()})) {
        const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_PNG);
        if (!codec) throw std::runtime_error("FFmpeg PNG encoder is unavailable");
        this->encoder_ = require(CodecPtr{avcodec_alloc_context3(codec)});
        const FrameSize size = working_size(source.width,source.height,max_edge);
        this->encoder_->width = size.width; this->encoder_->height = size.height;
        this->encoder_->thread_count = 1;
        this->encoder_->pix_fmt = AV_PIX_FMT_RGB24;
        this->encoder_->time_base = AVRational{1,1};
        this->encoder_->compression_level = 1;
        check(avcodec_open2(this->encoder_.get(),codec,nullptr), "Open PNG encoder");
    }
    void write(const AVFrame& source, const std::filesystem::path& path, std::size_t index) {
        AVFrame& rgb = this->converter_.convert(source);
        rgb.pts = static_cast<std::int64_t>(index);
        check(avcodec_send_frame(this->encoder_.get(),&rgb), "Encode PNG");
        check(avcodec_receive_packet(this->encoder_.get(),this->packet_.get()), "Receive PNG");
        std::ofstream output(path,std::ios::binary);
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output.write(reinterpret_cast<const char*>(this->packet_->data),this->packet_->size);
        output.close();
        av_packet_unref(this->packet_.get());
    }
private:
    RgbConverter converter_;
    CodecPtr encoder_;
    PacketPtr packet_;
};

// Each worker owns its encoder and RGB buffers. The caller bounds outstanding
// futures, so full-resolution frames cannot accumulate without limit.
class FrameWriterPool final {
public:
    explicit FrameWriterPool(int max_edge, unsigned worker_limit = 0) : max_edge_(max_edge) {
        try {
            const unsigned count = worker_limit == 0 ? this->worker_count() : worker_limit;
            for (unsigned index = 0; index < count; ++index)
                this->workers_.emplace_back([this] { this->work(); });
        } catch (...) { this->stop(); throw; }
    }
    ~FrameWriterPool() { this->stop(); }
    FrameWriterPool(const FrameWriterPool&) = delete;
    FrameWriterPool& operator=(const FrameWriterPool&) = delete;

    [[nodiscard]] std::size_t capacity() const noexcept { return this->workers_.size() * 2; }

    [[nodiscard]] std::future<void> submit(FramePtr frame, std::filesystem::path path, std::size_t index) {
        Job job{std::move(frame), std::move(path), index, {}};
        std::future<void> result = job.completion.get_future();
        {
            const std::lock_guard<std::mutex> lock(this->mutex_);
            this->jobs_.push_back(std::move(job));
        }
        this->available_.notify_one();
        return result;
    }

private:
    struct Job final {
        FramePtr frame;
        std::filesystem::path path;
        std::size_t index;
        std::promise<void> completion;
    };

    [[nodiscard]] unsigned worker_count() const noexcept {
        return std::clamp(std::thread::hardware_concurrency(), 1u, 8u);
    }
    void stop() noexcept {
        {
            const std::lock_guard<std::mutex> lock(this->mutex_);
            this->stopping_ = true;
            this->jobs_.clear();
        }
        this->available_.notify_all();
        for (std::thread& worker : this->workers_) if (worker.joinable()) worker.join();
    }
    void work() {
        std::unique_ptr<PngWriter> writer;
        while (true) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(this->mutex_);
                this->available_.wait(lock, [this] { return this->stopping_ || !this->jobs_.empty(); });
                if (this->stopping_) return;
                job = std::move(this->jobs_.front());
                this->jobs_.pop_front();
            }
            try {
                if (!writer) writer = std::make_unique<PngWriter>(*job.frame, this->max_edge_);
                writer->write(*job.frame, job.path, job.index);
                job.completion.set_value();
            } catch (...) { job.completion.set_exception(std::current_exception()); }
        }
    }
    int max_edge_;
    std::mutex mutex_;
    std::condition_variable available_;
    std::deque<Job> jobs_;
    std::vector<std::thread> workers_;
    bool stopping_{false};
};
}  // namespace sceneforge::video::detail
