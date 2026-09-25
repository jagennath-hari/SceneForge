#include "stereoforge/video/learned_features.hpp"
#include "stereoforge/video/keyframe_kernels.cuh"
#include "stereoforge/video/cuda_ransac.cuh"
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
    if (image.type() != CV_8UC3 || image.empty()) throw std::runtime_error("Expected BGR image");
    this->last_stream_ = stream;
    const std::size_t row_bytes = static_cast<std::size_t>(image.cols) * image.elemSize();
    const std::size_t size = image.total() * image.elemSize();
    if (!this->pixels_ || this->pixels_->bytes() < size) {
        check(cudaStreamSynchronize(stream));
        this->pixels_.reset();
        if (this->pinned_) { check(cudaFreeHost(this->pinned_)); this->pinned_ = nullptr; }
        check(cudaHostAlloc(reinterpret_cast<void**>(&this->pinned_), size, cudaHostAllocDefault));
        this->pixels_ = std::make_unique<DeviceBuffer>(this->device_, size);
    }
    for (int row = 0; row < image.rows; ++row)
        std::memcpy(this->pinned_ + static_cast<std::size_t>(row) * row_bytes, image.ptr(row), row_bytes);
    check(cudaMemcpyAsync(this->pixels_->data(), this->pinned_, size, cudaMemcpyHostToDevice, stream));
    return static_cast<const unsigned char*>(this->pixels_->data());
}
RaCoALIKEDExtractor::RaCoALIKEDExtractor(LearnedOptions options, int device) : options_(std::move(options)),
    runner_(this->options_.models / ("raco_aliked_" + std::to_string(device) + ".engine"), device),
    upload_(device), statistics_(device, 4 * sizeof(float)) {
    this->runner_.require("image", {1, 3, this->options_.height, this->options_.width}, true);
    this->runner_.require("keypoints", {1, this->options_.features, 2}, false);
    this->runner_.require("scores", {1, this->options_.features}, false);
    this->runner_.require("descriptors", {1, this->options_.features, 128}, false);
}
FrameFeatures RaCoALIKEDExtractor::extract(const std::filesystem::path& path, std::size_t index, double timestamp) {
    const cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("Cannot read candidate: " + path.string());
    const double scale = std::min({1.0, static_cast<double>(this->options_.width) / image.cols,
        static_cast<double>(this->options_.height) / image.rows});
    FrameFeatures result;
    result.index = index;
    result.timestamp = timestamp;
    result.size = cv::Size(std::max(1, static_cast<int>(std::round(image.cols * scale))),
                          std::max(1, static_cast<int>(std::round(image.rows * scale))));
    if (image.cols > 65536 || image.rows > 65536)
        throw std::runtime_error("Candidate dimensions exceed the supported bicubic resize range");
    const std::size_t resize_bytes = static_cast<std::size_t>(result.size.width) * image.rows * 3 * sizeof(float);
    if (!this->resize_workspace_ || this->resize_workspace_->bytes() < resize_bytes)
        this->resize_workspace_ = std::make_unique<DeviceBuffer>(this->runner_.device(), resize_bytes);
    for (const std::shared_ptr<DeviceFeatures>& slot : this->pool_) {
        if (slot.use_count() == 1) { result.device_features = slot; break; }
    }
    if (!result.device_features) {
        result.device_features = std::make_shared<DeviceFeatures>(this->runner_.device(),
            this->runner_.bytes("keypoints"), this->runner_.bytes("descriptors"), this->runner_.bytes("scores"));
        this->pool_.push_back(result.device_features);
    }
    result.device_features->invalidateMirror();
    this->runner_.bind("keypoints", result.device_features->points);
    this->runner_.bind("descriptors", result.device_features->descriptors);
    this->runner_.bind("scores", result.device_features->scores);
    const unsigned char* pixels = this->upload_.upload(image, this->runner_.stream());
    prepareRgbImage(pixels, image.cols, image.rows, this->runner_.address("image"), this->runner_.half("image"),
        result.size.width, result.size.height, this->options_.width, this->options_.height,
        static_cast<float*>(this->resize_workspace_->data()), this->resize_workspace_->bytes(),
        static_cast<float*>(this->statistics_.data()), this->runner_.stream());
    this->runner_.run();
    validateDescriptors(result.device_features->descriptors.data(), this->runner_.half("descriptors"),
        static_cast<std::size_t>(this->options_.features) * 128,
        static_cast<float*>(this->statistics_.data()), this->runner_.stream());
    validateKeypoints(result.device_features->points.data(), result.device_features->scores.data(),
        this->runner_.half("keypoints"), this->options_.features, result.size.width, result.size.height,
        this->options_.detector_threshold, static_cast<float*>(this->statistics_.data()), this->runner_.stream());
    float statistics[4]{};
    check(cudaMemcpyAsync(statistics, this->statistics_.data(), sizeof(statistics), cudaMemcpyDeviceToHost, this->runner_.stream()));
    this->runner_.finish();
    if (statistics[2] != 0) throw std::runtime_error("RaCo–ALIKED returned nonfinite features; compare an FP32 build");
    const double count = static_cast<double>(result.size.width) * result.size.height;
    const double mean = statistics[0] / count;
    result.sharpness = std::max(0.0, statistics[1] / count - mean * mean);
    result.feature_count = static_cast<std::size_t>(statistics[3]);
    // Only the optional OpenCV viewer needs individual keypoints on the CPU.
    if (this->options_.debug) {
        const std::vector<float> points = this->runner_.output("keypoints");
        const std::vector<float> scores = this->runner_.output("scores");
        for (int i = 0; i < this->options_.features; ++i) {
            const float x = points[2*i], y = points[2*i+1];
            const bool usable = x >= 4 && y >= 4 && x < result.size.width-4 && y < result.size.height-4 &&
                scores[i] >= this->options_.detector_threshold;
            result.points.emplace_back(x, y, 1.0f, -1.0f, usable ? scores[i] : -1.0f);
        }
    }
    return result;
}
LightGlueMatcher::LightGlueMatcher(LearnedOptions options, KeyframeOptions selection, int device) :
    options_(std::move(options)), selection_(std::move(selection)),
    runner_(this->options_.models / ("lightglue_" + std::to_string(device) + ".engine"), device),
    matches_(device, maximum_keypoints * sizeof(DeviceMatch)), match_count_(device, sizeof(int)),
    models_(device, ransac_models * sizeof(RansacModel)), winners_(device, 2 * sizeof(int)),
    mask_(device, maximum_keypoints * sizeof(unsigned char)), summary_(device, sizeof(RansacSummary)) {
    for (const std::string& suffix : {std::string("0"), std::string("1")}) {
        this->runner_.require("points" + suffix, {1, this->options_.features, 2}, true);
        this->runner_.require("descriptors" + suffix, {1, this->options_.features, 128}, true);
    }
    this->runner_.require("indices", {1, this->options_.features}, false, true);
    this->runner_.require("confidence", {1, this->options_.features}, false);
}
MatchQuality LightGlueMatcher::match(const FrameFeatures& reference, const FrameFeatures& candidate) {
    if (!reference.device_features || !candidate.device_features)
        throw std::runtime_error("LightGlue+ requires resident RaCo–ALIKED features");
    DeviceFeatures& a = reference.device_features->onDevice(this->runner_.device(), this->runner_.stream());
    DeviceFeatures& b = candidate.device_features->onDevice(this->runner_.device(), this->runner_.stream());
    this->runner_.bind("points0", a.points);
    this->runner_.bind("descriptors0", a.descriptors);
    this->runner_.bind("points1", b.points);
    this->runner_.bind("descriptors1", b.descriptors);
    this->runner_.run();
    // Matching and geometric verification share the same device allocations and
    // CUDA stream. No keypoints/indices/confidences are read back for RANSAC.
    verifyDeviceMatches(a.points.data(), a.scores.data(), b.points.data(), b.scores.data(),
        static_cast<const std::int32_t*>(this->runner_.address("indices")), this->runner_.address("confidence"),
        this->runner_.half("points0"), this->options_.features,
        reference.size.width, reference.size.height, candidate.size.width, candidate.size.height,
        this->options_.detector_threshold, this->options_.match_threshold,
        static_cast<float>(this->selection_.ransac_pixels),
        static_cast<unsigned>(reference.index*73856093u ^ candidate.index*19349663u ^ 0xa511e9b3u),
        static_cast<DeviceMatch*>(this->matches_.data()), static_cast<int*>(this->match_count_.data()),
        static_cast<RansacModel*>(this->models_.data()), static_cast<int*>(this->winners_.data()),
        static_cast<unsigned char*>(this->mask_.data()), static_cast<RansacSummary*>(this->summary_.data()),
        this->runner_.stream());
    RansacSummary summary{};
    check(cudaMemcpyAsync(&summary, this->summary_.data(), sizeof(summary), cudaMemcpyDeviceToHost, this->runner_.stream()));
    this->runner_.finish();
    if (summary.invalid) throw std::runtime_error("LightGlue+ returned invalid indices/confidence; compare FP32");
    MatchQuality result;
    result.matches = static_cast<std::size_t>(summary.matches);
    result.inliers = static_cast<std::size_t>(summary.inliers);
    result.homography = summary.homography != 0;
    result.ratio = summary.matches ? static_cast<double>(summary.inliers)/summary.matches : 0;
    result.coverage = summary.coverage;
    result.displacement = summary.displacement;
    if (this->options_.debug && result.matches) {
        std::vector<DeviceMatch> matches(result.matches);
        result.inlier_mask.resize(result.matches);
        check(cudaMemcpyAsync(matches.data(), this->matches_.data(), matches.size()*sizeof(DeviceMatch),
            cudaMemcpyDeviceToHost, this->runner_.stream()));
        check(cudaMemcpyAsync(result.inlier_mask.data(), this->mask_.data(), result.matches,
            cudaMemcpyDeviceToHost, this->runner_.stream()));
        this->runner_.finish();
        for (const DeviceMatch& match : matches)
            result.correspondences.emplace_back(match.source, match.target, 1.0f-match.confidence);
    }
    return result;
}
}  // namespace stereoforge::video
