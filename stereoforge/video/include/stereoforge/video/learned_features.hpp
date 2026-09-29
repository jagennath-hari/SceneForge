#pragma once

#include "stereoforge/video/keyframe_selector.hpp"
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <map>
#include <cstdint>
#include <mutex>

namespace stereoforge::video {
class DeviceBuffer final {
public:
    DeviceBuffer(int device, std::size_t bytes);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    [[nodiscard]] void* data() const { return this->data_; }
    [[nodiscard]] std::size_t bytes() const { return this->bytes_; }
    [[nodiscard]] int device() const { return this->device_; }
private:
    int device_;
    std::size_t bytes_;
    void* data_{};
};

class DeviceFeatures final {
public:
    DeviceFeatures(int device, std::size_t points_bytes, std::size_t descriptor_bytes, std::size_t score_bytes);
    [[nodiscard]] DeviceFeatures& onDevice(int device, cudaStream_t stream);
    void invalidateMirror() { this->mirror_valid_ = false; }
    DeviceBuffer points;
    DeviceBuffer descriptors;
    DeviceBuffer scores;
private:
    std::shared_ptr<DeviceFeatures> mirror_;
    bool mirror_valid_{false};
};

// One execution context per thread/device. Buffers may be bound directly to
// retained frame allocations; those owners must outlive inference completion.
class TensorRunner final {
public:
    TensorRunner(const std::filesystem::path& engine, int device);
    ~TensorRunner();
    TensorRunner(const TensorRunner&) = delete;
    TensorRunner& operator=(const TensorRunner&) = delete;
    [[nodiscard]] std::vector<float> output(const std::string& name);
    void run();
    void finish();
    void bind(const std::string& name, const DeviceBuffer& memory);
    [[nodiscard]] void* address(const std::string& name) const;
    [[nodiscard]] std::size_t bytes(const std::string& name) const;
    [[nodiscard]] bool half(const std::string& name) const;
    [[nodiscard]] cudaStream_t stream() const { return this->stream_; }
    [[nodiscard]] int device() const { return this->device_; }
    void require(const std::string& name, const std::vector<std::int64_t>& shape, bool input,
                 bool integer = false) const;
private:
    class Logger final : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* message) noexcept override;
    private:
        std::mutex mutex_;
    };
    [[nodiscard]] Logger& logger() const;
    void release() noexcept;
    int device_;
    cudaStream_t stream_{};
    cudaEvent_t completed_{};
    std::map<std::vector<std::uintptr_t>, cudaGraphExec_t> graphs_;
    bool graphs_supported_{true};
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    struct Buffer final {
        std::unique_ptr<DeviceBuffer> owned;
        void* bound{};
        std::size_t count{};
        bool input{};
        nvinfer1::DataType type{};
    };
    std::map<std::string, Buffer> buffers_;
};

struct LearnedOptions final {
    std::filesystem::path models;
    int width{960};
    int height{544};
    int features{1024};
    float detector_threshold{0.0f};
    bool debug{false};
    float match_threshold{0.1f};
};

class ImageUpload final {
public:
    explicit ImageUpload(int device) : device_(device) {}
    ~ImageUpload();
    ImageUpload(const ImageUpload&) = delete;
    ImageUpload& operator=(const ImageUpload&) = delete;
    [[nodiscard]] const unsigned char* upload(const cv::Mat& image, cudaStream_t stream);
private:
    int device_;
    unsigned char* pinned_{};
    cudaStream_t last_stream_{};
    std::unique_ptr<DeviceBuffer> pixels_;
};

class RaCoALIKEDExtractor final {
public:
    RaCoALIKEDExtractor(LearnedOptions options, int device);
    [[nodiscard]] FrameFeatures extract(const cv::Mat& image, std::size_t index, double timestamp);
    [[nodiscard]] FrameFeatures extract(const std::filesystem::path& path,
        std::size_t index, double timestamp);
private:
    LearnedOptions options_;
    TensorRunner runner_;
    ImageUpload upload_;
    DeviceBuffer statistics_;
    std::unique_ptr<DeviceBuffer> resize_workspace_;
    std::vector<std::shared_ptr<DeviceFeatures>> pool_;
};

class LightGlueMatcher final {
public:
    LightGlueMatcher(LearnedOptions options, KeyframeOptions selection, int device);
    [[nodiscard]] MatchQuality match(const FrameFeatures& reference, const FrameFeatures& candidate);
private:
    LearnedOptions options_;
    KeyframeOptions selection_;
    TensorRunner runner_;
    DeviceBuffer matches_;
    DeviceBuffer match_count_;
    DeviceBuffer models_;
    DeviceBuffer winners_;
    DeviceBuffer mask_;
    DeviceBuffer summary_;
};
}  // namespace stereoforge::video
