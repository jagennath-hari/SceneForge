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

#include "sceneforge/video/learned_features.hpp"
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace sceneforge::video {
namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("Keyframe CUDA: ") + cudaGetErrorString(error));
}
std::size_t elementBytes(nvinfer1::DataType type) {
    if (type == nvinfer1::DataType::kHALF) return 2;
    if (type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kINT32) return 4;
    throw std::runtime_error("Unsupported tensor type");
}
}
DeviceBuffer::DeviceBuffer(int device, std::size_t bytes) : device_(device), bytes_(bytes) {
    check(cudaSetDevice(this->device_));
    check(cudaMalloc(&this->data_, this->bytes_));
}
DeviceBuffer::~DeviceBuffer() {
    cudaSetDevice(this->device_);
    if (this->data_) cudaFree(this->data_);
}
DeviceFeatures::DeviceFeatures(int device, std::size_t points_bytes, std::size_t descriptor_bytes, std::size_t score_bytes) :
    points(device, points_bytes), descriptors(device, descriptor_bytes), scores(device, score_bytes) {}
DeviceFeatures& DeviceFeatures::onDevice(int device, cudaStream_t stream) {
    if (this->points.device() == device) return *this;
    if (!this->mirror_valid_ || !this->mirror_ || this->mirror_->points.device() != device) {
        int accessible = 0;
        check(cudaDeviceCanAccessPeer(&accessible, device, this->points.device()));
        if (!accessible) throw std::runtime_error("GPU peer access unavailable; use CUDA_VISIBLE_DEVICES=0 for resident matching");
        check(cudaSetDevice(device));
        const cudaError_t enabled = cudaDeviceEnablePeerAccess(this->points.device(), 0);
        if (enabled != cudaErrorPeerAccessAlreadyEnabled) check(enabled);
        else cudaGetLastError();
        if (!this->mirror_ || this->mirror_->points.device() != device)
            this->mirror_ = std::make_shared<DeviceFeatures>(device, this->points.bytes(), this->descriptors.bytes(), this->scores.bytes());
        check(cudaMemcpyPeerAsync(this->mirror_->points.data(), device, this->points.data(), this->points.device(), this->points.bytes(), stream));
        check(cudaMemcpyPeerAsync(this->mirror_->descriptors.data(), device, this->descriptors.data(), this->descriptors.device(), this->descriptors.bytes(), stream));
        check(cudaMemcpyPeerAsync(this->mirror_->scores.data(), device, this->scores.data(), this->scores.device(), this->scores.bytes(), stream));
        check(cudaStreamSynchronize(stream));
        this->mirror_valid_ = true;
    }
    return *this->mirror_;
}
void TensorRunner::Logger::log(Severity severity, const char* message) noexcept {
    if (severity > Severity::kWARNING) return;
    const std::lock_guard<std::mutex> lock(this->mutex_);
    std::cerr << "TensorRT: " << message << '\n';
}
TensorRunner::Logger& TensorRunner::logger() const {
    // TensorRT shares its logger across runtimes. One thread-safe instance must
    // outlive every worker's runtime, engine and execution context.
    static Logger shared_logger;
    return shared_logger;
}
TensorRunner::TensorRunner(const std::filesystem::path& path, int device) : device_(device) {
    try {
        check(cudaSetDevice(this->device_));
        check(cudaStreamCreateWithFlags(&this->stream_, cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&this->completed_, cudaEventDisableTiming));
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Missing TensorRT engine: " + path.string());
        const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        this->runtime_.reset(nvinfer1::createInferRuntime(this->logger()));
        if (!this->runtime_) throw std::runtime_error("Cannot create TensorRT runtime");
        this->engine_.reset(this->runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!this->engine_) throw std::runtime_error("Cannot load engine; regenerate models for this GPU/TensorRT version");
        this->context_.reset(this->engine_->createExecutionContext());
        if (!this->context_) throw std::runtime_error("Cannot create TensorRT execution context");
        for (int index = 0; index < this->engine_->getNbIOTensors(); ++index) {
            const char* name = this->engine_->getIOTensorName(index);
            const nvinfer1::DataType type = this->engine_->getTensorDataType(name);
            const std::size_t stride = elementBytes(type);
            if (this->engine_->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE ||
                this->engine_->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR)
                throw std::runtime_error("Expected linear device tensor: " + std::string(name));
            const nvinfer1::Dims shape = this->engine_->getTensorShape(name);
            std::size_t count = 1;
            if (shape.nbDims < 1) throw std::runtime_error("Expected static non-scalar tensor");
            for (int dimension = 0; dimension < shape.nbDims; ++dimension) {
                if (shape.d[dimension] <= 0 || static_cast<std::size_t>(shape.d[dimension]) >
                    std::numeric_limits<std::size_t>::max() / stride / count)
                    throw std::runtime_error("Expected bounded static tensor: " + std::string(name));
                count *= shape.d[dimension];
            }
            Buffer& buffer = this->buffers_[name];
            buffer.count = count;
            buffer.type = type;
            buffer.input = this->engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
            buffer.owned = std::make_unique<DeviceBuffer>(this->device_, count * stride);
            buffer.bound = buffer.owned->data();
            if (!this->context_->setTensorAddress(name, buffer.bound)) throw std::runtime_error("Cannot bind tensor");
        }
    } catch (...) { this->release(); throw; }
}
TensorRunner::~TensorRunner() { this->release(); }
void TensorRunner::release() noexcept {
    cudaSetDevice(this->device_);
    if (this->stream_) cudaStreamSynchronize(this->stream_);
    for (const std::pair<const std::vector<std::uintptr_t>, cudaGraphExec_t>& graph : this->graphs_)
        cudaGraphExecDestroy(graph.second);
    this->graphs_.clear();
    if (this->completed_) cudaEventDestroy(this->completed_);
    this->completed_ = nullptr;
    this->context_.reset();
    this->buffers_.clear();
    if (this->stream_) cudaStreamDestroy(this->stream_);
    this->stream_ = nullptr;
    this->engine_.reset();
    this->runtime_.reset();
}
void TensorRunner::require(const std::string& name, const std::vector<std::int64_t>& shape, bool input, bool integer) const {
    const Buffer& buffer = this->buffers_.at(name);
    const nvinfer1::Dims actual = this->engine_->getTensorShape(name.c_str());
    if (buffer.input != input || actual.nbDims != static_cast<int>(shape.size()) ||
        (integer ? buffer.type != nvinfer1::DataType::kINT32 :
         (buffer.type != nvinfer1::DataType::kHALF && buffer.type != nvinfer1::DataType::kFLOAT)))
        throw std::runtime_error("Model tensor contract mismatch: " + name);
    for (int index = 0; index < actual.nbDims; ++index)
        if (actual.d[index] != shape[index]) throw std::runtime_error("Model shape mismatch: " + name);
}
void TensorRunner::bind(const std::string& name, const DeviceBuffer& memory) {
    check(cudaSetDevice(this->device_));
    if (memory.device() != this->device_ || memory.bytes() != this->bytes(name)) throw std::runtime_error("Incompatible device binding");
    if (!this->context_->setTensorAddress(name.c_str(), memory.data())) throw std::runtime_error("Cannot bind retained tensor");
    this->buffers_.at(name).bound = memory.data();
}
void* TensorRunner::address(const std::string& name) const { return this->buffers_.at(name).bound; }
std::size_t TensorRunner::bytes(const std::string& name) const {
    const Buffer& buffer = this->buffers_.at(name);
    return buffer.count * elementBytes(buffer.type);
}
bool TensorRunner::half(const std::string& name) const { return this->buffers_.at(name).type == nvinfer1::DataType::kHALF; }
void TensorRunner::finish() {
    check(cudaSetDevice(this->device_));
    check(cudaEventRecord(this->completed_, this->stream_));
    cudaError_t status;
    do { status = cudaEventQuery(this->completed_); } while (status == cudaErrorNotReady);
    check(status);  // Deliberate spin waiting, like trtexec --useSpinWait.
}
void TensorRunner::run() {
    check(cudaSetDevice(this->device_));
    std::vector<std::uintptr_t> key;
    for (const std::pair<const std::string, Buffer>& tensor : this->buffers_)
        key.push_back(reinterpret_cast<std::uintptr_t>(tensor.second.bound));
    const std::map<std::vector<std::uintptr_t>, cudaGraphExec_t>::const_iterator found = this->graphs_.find(key);
    if (found != this->graphs_.end()) {
        check(cudaGraphLaunch(found->second, this->stream_));
        return;
    }
    if (!this->context_->enqueueV3(this->stream_)) throw std::runtime_error("TensorRT inference failed");
    if (!this->graphs_supported_) return;
    // Warm inference above also serves this frame. Capture does not execute a
    // second inference. Retained pool slots make these address tuples reusable.
    this->finish();
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaError_t status = cudaStreamBeginCapture(this->stream_, cudaStreamCaptureModeThreadLocal);
    if (status == cudaSuccess) {
        const bool enqueued = this->context_->enqueueV3(this->stream_);
        status = cudaStreamEndCapture(this->stream_, &graph);
        if (!enqueued && status == cudaSuccess) status = cudaErrorStreamCaptureInvalidated;
    }
    if (status == cudaSuccess) status = cudaGraphInstantiate(&executable, graph, 0);
    if (graph) cudaGraphDestroy(graph);
    if (status != cudaSuccess) {
        if (executable) cudaGraphExecDestroy(executable);
        this->graphs_supported_ = false;
        cudaGetLastError();
        std::cerr << "CUDA graph capture unavailable; continuing with ordinary TensorRT enqueue\n";
        return;
    }
    // Bound graph metadata even when keyframes keep changing the address pairs.
    if (this->graphs_.size() >= 32) {
        cudaGraphExecDestroy(this->graphs_.begin()->second);
        this->graphs_.erase(this->graphs_.begin());
    }
    this->graphs_.emplace(std::move(key), executable);
}
std::vector<float> TensorRunner::output(const std::string& name) {
    check(cudaSetDevice(this->device_));
    const Buffer& buffer = this->buffers_.at(name);
    if (buffer.input || buffer.type == nvinfer1::DataType::kINT32) throw std::runtime_error("Expected floating output: " + name);
    std::vector<float> result(buffer.count);
    if (buffer.type == nvinfer1::DataType::kHALF) {
        std::vector<__half> staging(buffer.count);
        check(cudaMemcpyAsync(staging.data(), buffer.bound, this->bytes(name), cudaMemcpyDeviceToHost, this->stream_));
        this->finish();
        for (std::size_t index = 0; index < buffer.count; ++index) result[index] = __half2float(staging[index]);
    } else {
        check(cudaMemcpyAsync(result.data(), buffer.bound, this->bytes(name), cudaMemcpyDeviceToHost, this->stream_));
        this->finish();
    }
    return result;
}
}  // namespace sceneforge::video
