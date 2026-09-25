#include "stereoforge/video/keyframe_selector.hpp"
#include "stereoforge/video/learned_features.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <exception>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace stereoforge::video {
void KeyframeOptions::validate() const {
    if (this->features < 120 || this->features > 4096 || this->workers == 0 || this->min_features < 8 ||
        this->min_features > this->features || this->min_inliers < 8 || this->min_inliers > this->features)
        throw std::invalid_argument("Invalid keyframe feature counts or worker count");
    for (double value : {this->min_inlier_ratio, this->min_coverage, this->retention, this->motion})
        if (!std::isfinite(value) || value <= 0 || value >= 1) throw std::invalid_argument("Keyframe ratios must be between zero and one");
    for (double value : {this->ransac_pixels, this->tracking_timeout})
        if (!std::isfinite(value) || value <= 0) throw std::invalid_argument("Keyframe thresholds must be positive");
    for (double value : {this->min_sharpness, this->min_interval})
        if (!std::isfinite(value) || value < 0) throw std::invalid_argument("Keyframe thresholds must be nonnegative");
}
namespace {
class OrderedFeatures final {
public:
    OrderedFeatures(const std::filesystem::path& directory, const nlohmann::json& frames, ExtractorFactory factory, unsigned workers) :
        directory_(directory), frames_(frames), factory_(std::move(factory)),
        capacity_(2 * workers) {
        try {
            for (unsigned index = 0; index < this->capacity_ / 2; ++index)
                this->workers_.emplace_back([this] { this->work(); });
        } catch (...) { this->stop(); throw; }
    }
    ~OrderedFeatures() { this->stop(); }
    OrderedFeatures(const OrderedFeatures&) = delete;
    OrderedFeatures& operator=(const OrderedFeatures&) = delete;
    [[nodiscard]] FrameFeatures next() {
        std::unique_lock<std::mutex> lock(this->mutex_);
        this->changed_.wait(lock, [this] { return this->error_ || this->ready_.contains(this->consumed_); });
        if (this->error_) std::rethrow_exception(this->error_);
        FrameFeatures result = std::move(this->ready_.at(this->consumed_));
        this->ready_.erase(this->consumed_++);
        this->changed_.notify_all();
        return result;
    }
private:
    void stop() noexcept {
        { const std::lock_guard<std::mutex> lock(this->mutex_); this->stopped_ = true; }
        this->changed_.notify_all();
        for (std::thread& thread : this->workers_) if (thread.joinable()) thread.join();
    }
    void work() {
        try {
            std::unique_ptr<RaCoALIKEDExtractor> extractor = this->factory_();
            if (!extractor) throw std::runtime_error("Feature extractor factory returned null");
            while (true) {
                std::size_t index;
                {
                    std::unique_lock<std::mutex> lock(this->mutex_);
                    this->changed_.wait(lock, [this] {
                        return this->stopped_ || this->error_ || this->assigned_ >= this->frames_.size() ||
                               this->assigned_ < this->consumed_ + this->capacity_;
                    });
                    if (this->stopped_ || this->error_ || this->assigned_ >= this->frames_.size()) return;
                    index = this->assigned_++;
                }
                const nlohmann::json& record = this->frames_.at(index);
                FrameFeatures features = extractor->extract(this->directory_ / record.at("file").get<std::string>(),
                    index, record.at("timestamp_seconds").get<double>());
                { const std::lock_guard<std::mutex> lock(this->mutex_); this->ready_.emplace(index, std::move(features)); }
                this->changed_.notify_all();
            }
        } catch (...) {
            { const std::lock_guard<std::mutex> lock(this->mutex_); this->error_ = std::current_exception(); }
            this->changed_.notify_all();
        }
    }
    std::filesystem::path directory_;
    const nlohmann::json& frames_;
    ExtractorFactory factory_;
    std::size_t capacity_;
    std::size_t assigned_{};
    std::size_t consumed_{};
    std::mutex mutex_;
    std::condition_variable changed_;
    std::map<std::size_t, FrameFeatures> ready_;
    std::vector<std::thread> workers_;
    std::exception_ptr error_;
    bool stopped_{false};
};
}
KeyframeSelector::KeyframeSelector(KeyframeOptions options, ExtractorFactory factory, std::unique_ptr<LightGlueMatcher> matcher) :
    options_(options), factory_(std::move(factory)), matcher_(std::move(matcher)) {
    this->options_.validate();
    if (!this->factory_ || !this->matcher_)
        throw std::invalid_argument("RaCo extractor factory and LightGlue matcher are required");
}

KeyframeSelector::~KeyframeSelector() = default;

bool KeyframeSelector::run(const std::filesystem::path& directory, const std::filesystem::path& output,
                           const ProgressCallback& progress, const DebugCallback& debug) {
    std::ifstream input(directory / "manifest.json");
    input.exceptions(std::ios::failbit | std::ios::badbit);
    nlohmann::json manifest;
    input >> manifest;
    const nlohmann::json frames = manifest.at("frames");
    if (frames.empty()) throw std::runtime_error("No frames to select");
    double previous_time = -1e100;
    for (const nlohmann::json& frame : frames) {
        const std::filesystem::path name(frame.at("file").get<std::string>());
        const double timestamp = frame.at("timestamp_seconds").get<double>();
        if (name.has_parent_path() || name.empty() || !std::isfinite(timestamp) || timestamp <= previous_time)
            throw std::runtime_error("Invalid candidate paths or timestamps");
        previous_time = timestamp;
    }
    OrderedFeatures features(directory, frames, this->factory_, this->options_.workers);
    std::optional<FrameFeatures> anchor, last_good;
    std::size_t best_inliers = 0;
    nlohmann::json selected = nlohmann::json::array();
    nlohmann::json decisions = nlohmann::json::array();
    bool tracking_break = false;
    bool interrupted = false;
    const std::function<bool(const MatchQuality&)> reliable = [this](const MatchQuality& quality) {
        return quality.inliers >= static_cast<std::size_t>(this->options_.min_inliers) &&
               quality.ratio >= this->options_.min_inlier_ratio && quality.coverage >= this->options_.min_coverage;
    };
    const std::function<void(const FrameFeatures&, const char*)> accept = [&](const FrameFeatures& frame, const char* reason) {
        nlohmann::json record = frames.at(frame.index);
        record["candidate_index"] = frame.index;
        record["source_frame_index"] = record.value("source_frame_index", frame.index);
        record["selection_reason"] = reason;
        selected.push_back(std::move(record));
        anchor = frame;
        best_inliers = 0;
    };
    if (progress) progress(0, frames.size(), 0);
    for (std::size_t index = 0; index < frames.size(); ++index) {
        FrameFeatures current = features.next();
        std::optional<FrameFeatures> reference;
        if (debug) reference = anchor;
        MatchQuality quality;
        nlohmann::json decision{{"candidate_index", index}, {"timestamp_seconds", current.timestamp},
            {"features", current.feature_count}, {"sharpness", current.sharpness}};
        const bool usable = current.feature_count >= static_cast<std::size_t>(this->options_.min_features) &&
                            current.sharpness >= this->options_.min_sharpness;
        if (!usable) decision["decision"] = "reject_quality";
        else if (!anchor) {
            accept(current, "first_usable");
            last_good = current;
            decision["decision"] = "first_usable";
        } else {
            quality = this->matcher_->match(*anchor, current);
            if (!reliable(quality) && last_good && last_good->index > anchor->index) {
                accept(*last_good, "before_tracking_loss");
                decision["bridge_candidate_index"] = last_good->index;
                quality = this->matcher_->match(*anchor, current);
            }
            if (debug) reference = anchor;
            decision["reference_candidate_index"] = anchor->index;
            decision["matches"] = quality.matches;
            decision["inliers"] = quality.inliers;
            decision["inlier_ratio"] = quality.ratio;
            decision["coverage"] = quality.coverage;
            decision["displacement"] = quality.displacement;
            decision["model"] = quality.homography ? "homography" : "fundamental";
            if (reliable(quality)) {
                best_inliers = std::max(best_inliers, quality.inliers);
                const bool overlap_drop = quality.inliers < this->options_.retention * best_inliers;
                const bool moved = quality.displacement >= this->options_.motion;
                if (current.timestamp - anchor->timestamp >= this->options_.min_interval && (overlap_drop || moved)) {
                    accept(current, overlap_drop ? "overlap_drop" : "image_motion");
                    decision["decision"] = overlap_drop ? "overlap_drop" : "image_motion";
                } else decision["decision"] = "redundant";
                last_good = current;
            } else decision["decision"] = "unreliable_matches";
        }
        if (last_good && current.timestamp - last_good->timestamp > this->options_.tracking_timeout) {
            tracking_break = true;
            decision["decision"] = "tracking_break";
        }
        if (debug && !debug(reference ? &*reference : nullptr, current, quality,
                            decision.at("decision").get<std::string>(), selected.size())) interrupted = true;
        decisions.push_back(std::move(decision));
        if (progress) progress(index + 1, frames.size(), selected.size());
        if (tracking_break || interrupted) break;
    }
    if (!interrupted && !tracking_break && last_good && anchor && last_good->index != anchor->index) accept(*last_good, "last_usable");
    const bool success = !interrupted && !tracking_break && selected.size() >= 3;
    if (progress) progress(decisions.size(), frames.size(), selected.size());
    manifest["candidate_frame_count"] = frames.size();
    manifest["frames"] = std::move(selected);
    manifest["keyframe_selection"] = {{"status", interrupted ? "interrupted" : (tracking_break ? "tracking_break" : (success ? "complete" : "insufficient_keyframes"))},
        {"frontend", keyframe_label}, {"decisions", std::move(decisions)}};
    const std::filesystem::path temporary = output.string() + ".tmp";
    std::ofstream report(temporary);
    report.exceptions(std::ios::badbit | std::ios::failbit);
    report << manifest.dump(2) << '\n';
    report.close();
    std::filesystem::rename(temporary, output);
    return success;
}
}  // namespace stereoforge::video
