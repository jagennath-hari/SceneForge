#include "stereoforge/video/learned_features.hpp"
#include "stereoforge/video/keyframe_kernels.cuh"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace stereoforge::video {
namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("Keyframe CUDA: ") + cudaGetErrorString(error));
}
}
ImageUpload::~ImageUpload() {
    cudaSetDevice(this->device_);
    if (this->last_stream_) cudaStreamSynchronize(this->last_stream_);
    // The device allocation is released before its asynchronous upload source.
    this->pixels_.reset();
    if (this->pinned_) cudaFreeHost(this->pinned_);
}
const unsigned char* ImageUpload::upload(const cv::Mat& image, cudaStream_t stream) {
    check(cudaSetDevice(this->device_));
    if (image.type() != CV_8UC1 || image.empty()) throw std::runtime_error("Expected grayscale image");
    this->last_stream_ = stream;
    const std::size_t size = image.total();
    if (!this->pixels_ || this->pixels_->bytes() < size) {
        check(cudaStreamSynchronize(stream));
        this->pixels_.reset();
        if (this->pinned_) { check(cudaFreeHost(this->pinned_)); this->pinned_ = nullptr; }
        check(cudaHostAlloc(reinterpret_cast<void**>(&this->pinned_), size, cudaHostAllocDefault));
        this->pixels_ = std::make_unique<DeviceBuffer>(this->device_, size);
    }
    for (int row = 0; row < image.rows; ++row)
        std::memcpy(this->pinned_ + static_cast<std::size_t>(row) * image.cols, image.ptr(row), image.cols);
    check(cudaMemcpyAsync(this->pixels_->data(), this->pinned_, size, cudaMemcpyHostToDevice, stream));
    return static_cast<const unsigned char*>(this->pixels_->data());
}
SuperPointExtractor::SuperPointExtractor(LearnedOptions options, int device) : options_(std::move(options)),
    runner_(this->options_.models / ("superpoint_" + std::to_string(device) + ".engine"), device),
    upload_(device), statistics_(device, 3 * sizeof(float)) {
    this->runner_.require("image", {1, 1, this->options_.height, this->options_.width}, true);
    this->runner_.require("keypoints", {1, this->options_.features, 2}, false);
    this->runner_.require("scores", {1, this->options_.features}, false);
    this->runner_.require("descriptors", {1, this->options_.features, 256}, false);
}
FrameFeatures SuperPointExtractor::extract(const std::filesystem::path& path, std::size_t index, double timestamp) {
    const cv::Mat gray = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
    if (gray.empty()) throw std::runtime_error("Cannot read candidate: " + path.string());
    const double scale = std::min({1.0, static_cast<double>(this->options_.width) / gray.cols,
        static_cast<double>(this->options_.height) / gray.rows});
    FrameFeatures result;
    result.index = index;
    result.timestamp = timestamp;
    result.size = cv::Size(std::max(1, static_cast<int>(std::round(gray.cols * scale))),
                          std::max(1, static_cast<int>(std::round(gray.rows * scale))));
    for (const std::shared_ptr<DeviceFeatures>& slot : this->pool_) {
        if (slot.use_count() == 1) { result.device_features = slot; break; }
    }
    if (!result.device_features) {
        result.device_features = std::make_shared<DeviceFeatures>(this->runner_.device(),
            this->runner_.bytes("keypoints"), this->runner_.bytes("descriptors"));
        this->pool_.push_back(result.device_features);
    }
    result.device_features->invalidateMirror();
    this->runner_.bind("keypoints", result.device_features->points);
    this->runner_.bind("descriptors", result.device_features->descriptors);
    const unsigned char* pixels = this->upload_.upload(gray, this->runner_.stream());
    prepareGrayImage(pixels, gray.cols, gray.rows, this->runner_.address("image"), this->runner_.half("image"),
        result.size.width, result.size.height, this->options_.width, this->options_.height,
        static_cast<float*>(this->statistics_.data()), this->runner_.stream());
    this->runner_.run();
    validateDescriptors(result.device_features->descriptors.data(), this->runner_.half("descriptors"),
        static_cast<std::size_t>(this->options_.features) * 256,
        static_cast<float*>(this->statistics_.data()), this->runner_.stream());
    // Only small keypoint/score metadata is downloaded for RANSAC and the debugger.
    // Descriptor tensors stay on the device and retain their original output allocation.
    const std::vector<float> points = this->runner_.output("keypoints");
    const std::vector<float> scores = this->runner_.output("scores");
    float statistics[3]{};
    check(cudaMemcpyAsync(statistics, this->statistics_.data(), sizeof(statistics), cudaMemcpyDeviceToHost, this->runner_.stream()));
    this->runner_.finish();
    if (statistics[2] != 0) throw std::runtime_error("SuperPoint returned nonfinite descriptors; compare an FP32 build");
    const double count = static_cast<double>(result.size.width) * result.size.height;
    const double mean = statistics[0] / count;
    result.sharpness = std::max(0.0, statistics[1] / count - mean * mean);
    for (int position = 0; position < this->options_.features; ++position) {
        const float x = points[2 * position], y = points[2 * position + 1], score = scores[position];
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(score))
            throw std::runtime_error("SuperPoint returned nonfinite keypoints/scores");
        const bool usable = x >= 4 && y >= 4 && x < result.size.width - 4 && y < result.size.height - 4 &&
                            score >= this->options_.detector_threshold;
        result.points.emplace_back(x, y, 1.0f, -1.0f, usable ? score : -1.0f);
        if (usable) ++result.feature_count;
    }
    return result;
}
LightGlueMatcher::LightGlueMatcher(LearnedOptions options, KeyframeOptions selection, int device) :
    options_(std::move(options)), selection_(std::move(selection)),
    runner_(this->options_.models / ("lightglue_" + std::to_string(device) + ".engine"), device) {
    for (const std::string& suffix : {std::string("0"), std::string("1")}) {
        this->runner_.require("points" + suffix, {1, this->options_.features, 2}, true);
        this->runner_.require("descriptors" + suffix, {1, this->options_.features, 256}, true);
    }
    this->runner_.require("indices", {1, this->options_.features}, false, true);
    this->runner_.require("confidence", {1, this->options_.features}, false);
}
MatchQuality LightGlueMatcher::match(const FrameFeatures& reference, const FrameFeatures& candidate) {
    int side = 0;
    for (const FrameFeatures* frame : {&reference, &candidate}) {
        if (!frame->device_features || frame->points.size() != static_cast<std::size_t>(this->options_.features))
            throw std::runtime_error("LightGlue requires resident SuperPoint features");
        DeviceFeatures& features = frame->device_features->onDevice(this->runner_.device(), this->runner_.stream());
        const std::string suffix = std::to_string(side++);
        this->runner_.bind("points" + suffix, features.points);
        this->runner_.bind("descriptors" + suffix, features.descriptors);
    }
    this->runner_.run();
    const std::vector<std::int32_t> indices = this->runner_.indices("indices");
    const std::vector<float> confidence = this->runner_.output("confidence");
    std::vector<cv::DMatch> matches;
    for (int index = 0; index < this->options_.features; ++index) {
        if (!std::isfinite(confidence[index])) throw std::runtime_error("LightGlue returned nonfinite confidence; compare FP32");
        const int target = indices[index];
        if (target == -1) continue;
        if (target < 0 || target >= this->options_.features) throw std::runtime_error("Invalid LightGlue match index");
        if (confidence[index] < this->options_.match_threshold || reference.points[index].response < 0 ||
            candidate.points[target].response < 0) continue;
        matches.emplace_back(index, target, 1.0f - confidence[index]);
    }
    return verifyMatches(reference, candidate, std::move(matches), this->selection_);
}
}  // namespace stereoforge::video
