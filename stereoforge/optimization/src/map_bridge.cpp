#include "stereoforge/optimization/map_bridge.hpp"
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>

namespace stereoforge::optimization {
namespace {
constexpr std::size_t kMinimumTraining = 30;
constexpr std::size_t kMinimumHoldouts = 10;
constexpr std::size_t kMaximumSolves = 16;
constexpr std::size_t kMaximumBridges = 4;

struct Correspondence {
    TrackId track;
    Eigen::Vector3d position;
    Eigen::Vector2d pixel;
    bool held_out;
};
struct Target {
    FrameId frame;
    std::vector<Correspondence> observations;
};

bool HeldOut(TrackId track) {
    // Stable track-level split, shared by all cameras and all proposals.
    std::uint64_t value = static_cast<std::uint64_t>(track);
    value = (value ^ (value >> 30))*0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27))*0x94d049bb133111ebULL;
    return ((value ^ (value >> 31)) % 5) == 0;
}
double Median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    return n % 2 ? values[n/2] : (values[n/2-1]+values[n/2])/2;
}
Camera Apply(const Camera& camera, const Similarity& transform) {
    Camera result = camera;
    result.rotation = transform.rotation*camera.rotation;
    result.center = transform.scale*transform.rotation*camera.center+transform.translation;
    return result;
}
bool Supported(const Camera& camera, const Target& target) {
    std::size_t training = 0, held_out = 0, passed = 0;
    for (const Correspondence& observation : target.observations) {
        const double error = Reprojection(camera, observation.position, observation.pixel);
        if (observation.held_out) {
            ++held_out;
            passed += error <= 5;
        } else {
            training += error <= 3;
        }
    }
    return training >= kMinimumTraining && held_out >= kMinimumHoldouts &&
        static_cast<double>(passed) >= .8*held_out;
}

class InitializationRecovery final {
public:
    InitializationRecovery(const std::vector<Observations>& tracks,
        const std::vector<DepthFrame>& frames, const SparseMap& accepted,
        const std::function<void(const std::string&)>& progress)
        : tracks_(tracks), accepted_(accepted), progress_(progress) {
        for (const DepthFrame& frame : frames) { this->frames_.emplace(frame.id, &frame); }
    }

    void Run(SparseMap& model) {
        std::vector<SparseMap> groups = ConnectedGroups(model);
        if (groups.size() < 2) { return; }
        // Prefer the component with most shared accepted cameras; otherwise the
        // largest surviving landmark set supplies the provisional local gauge.
        std::size_t anchor = 0;
        for (std::size_t i = 1; i < groups.size(); ++i) {
            if (this->Rank(groups[i]) > this->Rank(groups[anchor])) { anchor = i; }
        }
        SparseMap reference = std::move(groups[anchor]);
        groups.erase(groups.begin()+static_cast<std::ptrdiff_t>(anchor));
        std::size_t repaired = 0;
        this->progress_("initialization bridge recovery: " + std::to_string(groups.size()+1) + " components");
        while (repaired < kMaximumBridges && this->solves_ < kMaximumSolves) {
            bool advanced = false;
            for (std::size_t i = 0; i < groups.size() && this->solves_ < kMaximumSolves; ++i) {
                const std::vector<Target> targets = this->Targets(reference, groups[i]);
                if (targets.empty()) { continue; }
                std::string reason = "no supported pose/scale proposal";
                for (const Target& target : targets) {
                    if (this->solves_ >= kMaximumSolves) { break; }
                    ++this->solves_;
                    this->progress_("initialization bridge PnP/RANSAC: frame " + std::to_string(target.frame));
                    const std::optional<Camera> pose = this->Pose(groups[i].cameras.at(target.frame), target);
                    if (!pose) { reason = "PnP training/held-out support failed"; continue; }
                    const std::optional<Similarity> transform = this->Fit(groups[i], target, *pose);
                    if (!transform) { reason = "depth-scale support was inconsistent"; continue; }
                    std::size_t validated = 0;
                    bool valid = true;
                    for (const Target& check : targets) {
                        if (!Supported(Apply(groups[i].cameras.at(check.frame), *transform), check)) {
                            valid = false; break;
                        }
                        ++validated;
                    }
                    if (!valid || validated < std::min<std::size_t>(2, groups[i].cameras.size())) {
                        reason = "transformed component failed cross-camera validation"; continue;
                    }
                    SparseMap trial = reference;
                    SparseMap moved = groups[i];
                    Transform(moved, *transform);
                    trial.cameras.merge(moved.cameras);
                    trial.landmarks.merge(moved.landmarks);
                    const std::size_t bridges = this->Restore(trial, reference, groups[i]);
                    if (bridges < kMinimumTraining || ConnectedGroups(trial).size() != 1) {
                        reason = "fewer than 30 restored crossing tracks or disconnected proposal"; continue;
                    }
                    reference = std::move(trial);
                    groups.erase(groups.begin()+static_cast<std::ptrdiff_t>(i));
                    ++repaired;
                    this->progress_("initialization bridge accepted at frame " + std::to_string(target.frame) +
                        ": " + std::to_string(bridges) + " measured tracks; " + std::to_string(validated) + " checked cameras; continuing to local BA");
                    advanced = true;
                    break;
                }
                if (advanced) { break; }
                this->progress_("WARNING: Initialization bridge not recovered at frame " +
                    std::to_string(targets.front().frame) + ": " + reason);
            }
            if (!advanced) { break; }
        }
        if (repaired != 0) {
            for (SparseMap& group : groups) {
                reference.cameras.merge(group.cameras);
                reference.landmarks.merge(group.landmarks);
            }
            model = std::move(reference);
        }
        if (!groups.empty()) {
            this->progress_("WARNING: Initialization bridge recovery left " +
                std::to_string(groups.size()) + " disconnected component(s); normal window recovery remains active");
        }
    }

private:
    [[nodiscard]] std::pair<std::size_t, std::size_t> Rank(const SparseMap& group) const {
        std::size_t shared = 0;
        for (const std::pair<const FrameId, Camera>& camera : group.cameras) {
            shared += this->accepted_.cameras.contains(camera.first);
        }
        return {shared, group.landmarks.size()};
    }

    [[nodiscard]] std::vector<Target> Targets(const SparseMap& reference, const SparseMap& group) const {
        std::map<FrameId, std::vector<Correspondence>> candidates;
        for (const std::pair<const TrackId, Landmark>& entry : reference.landmarks) {
            // Only locally consistent 3D anchors may initialize another camera.
            std::size_t support = 0;
            for (const std::pair<const FrameId, Eigen::Vector2d>& observation : entry.second.observations) {
                support += Reprojection(reference.cameras.at(observation.first), entry.second.position, observation.second) <= 3;
            }
            if (support < 3) { continue; }
            for (const std::pair<const FrameId, Eigen::Vector2d>& observation : this->tracks_.at(entry.first)) {
                if (!group.cameras.contains(observation.first)) { continue; }
                const Camera& camera = group.cameras.at(observation.first);
                if (observation.second.x() < 0 || observation.second.x() >= camera.width ||
                    observation.second.y() < 0 || observation.second.y() >= camera.height) { continue; }
                candidates[observation.first].push_back({entry.first, entry.second.position,
                    observation.second, HeldOut(entry.first)});
            }
        }
        std::vector<Target> result;
        for (std::pair<const FrameId, std::vector<Correspondence>>& entry : candidates) {
            std::size_t held_out = 0;
            for (const Correspondence& observation : entry.second) { held_out += observation.held_out; }
            if (held_out >= kMinimumHoldouts && entry.second.size()-held_out >= kMinimumTraining) {
                result.push_back({entry.first, std::move(entry.second)});
            }
        }
        std::sort(result.begin(), result.end(), [](const Target& a, const Target& b) {
            if (a.observations.size() != b.observations.size()) { return a.observations.size() > b.observations.size(); }
            return a.frame < b.frame;
        });
        if (result.size() > 4) { result.resize(4); }
        return result;
    }

    [[nodiscard]] std::optional<Camera> Pose(const Camera& initial, const Target& target) const {
        std::vector<cv::Point3d> points;
        std::vector<cv::Point2d> pixels;
        for (const Correspondence& observation : target.observations) {
            if (observation.held_out) { continue; }
            points.emplace_back(observation.position.x(), observation.position.y(), observation.position.z());
            pixels.emplace_back(observation.pixel.x(), observation.pixel.y());
        }
        cv::Mat calibration(3,3,CV_64F);
        for (int r = 0; r < 3; ++r) { for (int c = 0; c < 3; ++c) { calibration.at<double>(r,c) = initial.intrinsics(r,c); } }
        cv::Mat rotation_vector(3,1,CV_64F), translation(3,1,CV_64F), inliers;
        try {
            if (!cv::solvePnPRansac(points, pixels, calibration, cv::noArray(), rotation_vector,
                translation, false, 500, 3.0f, .999, inliers, cv::SOLVEPNP_EPNP) ||
                inliers.total() < kMinimumTraining) { return std::nullopt; }
            std::vector<cv::Point3d> fitting_points;
            std::vector<cv::Point2d> fitting_pixels;
            for (int i = 0; i < inliers.rows; ++i) {
                const std::size_t index = static_cast<std::size_t>(inliers.at<int>(i,0));
                fitting_points.push_back(points.at(index)); fitting_pixels.push_back(pixels.at(index));
            }
            cv::solvePnPRefineLM(fitting_points, fitting_pixels, calibration, cv::noArray(), rotation_vector, translation);
            cv::Mat rotation;
            cv::Rodrigues(rotation_vector, rotation);
            Camera camera = initial;
            Eigen::Matrix3d world_to_camera;
            Eigen::Vector3d offset;
            for (int r = 0; r < 3; ++r) {
                offset[r] = translation.at<double>(r,0);
                for (int c = 0; c < 3; ++c) { world_to_camera(r,c) = rotation.at<double>(r,c); }
            }
            camera.rotation = world_to_camera.transpose();
            camera.center = -camera.rotation*offset;
            if (!camera.rotation.allFinite() || !camera.center.allFinite() || !Supported(camera, target)) { return std::nullopt; }
            return camera;
        } catch (const cv::Exception&) {
            return std::nullopt;
        }
    }

    [[nodiscard]] std::optional<Similarity> Fit(const SparseMap& group, const Target& target, const Camera& pose) const {
        const Camera& previous = group.cameras.at(target.frame);
        Similarity transform;
        if (group.cameras.size() > 1) {
            // PnP gives translation in the reference gauge. Calibrate the moving
            // group's depth scale on training points only, before transforming it.
            const DepthFrame& frame = *this->frames_.at(target.frame);
            std::vector<double> scales;
            for (const Correspondence& observation : target.observations) {
                if (observation.held_out || Reprojection(pose, observation.position, observation.pixel) > 3) { continue; }
                const int x = std::clamp(static_cast<int>(std::nearbyint(observation.pixel.x())), 0, frame.camera.width-1);
                const int y = std::clamp(static_cast<int>(std::nearbyint(observation.pixel.y())), 0, frame.camera.height-1);
                const double raw = frame.depth[static_cast<std::size_t>(y)*frame.camera.width+x];
                const double depth = (pose.rotation.transpose()*(observation.position-pose.center)).z();
                if (std::isfinite(raw) && raw > 0 && std::isfinite(depth) && depth > 0) { scales.push_back(std::log(depth/raw)); }
            }
            if (scales.size() < kMinimumTraining) { return std::nullopt; }
            const double median = Median(scales);
            std::size_t consistent = 0;
            for (const double value : scales) { consistent += std::abs(value-median) <= std::log(1.25); }
            if (static_cast<double>(consistent) < .7*scales.size()) { return std::nullopt; }
            transform.scale = std::exp(median);
        }
        transform.rotation = pose.rotation*previous.rotation.transpose();
        transform.translation = pose.center-transform.scale*transform.rotation*previous.center;
        if (!std::isfinite(transform.scale) || transform.scale <= 0 || !transform.translation.allFinite()) { return std::nullopt; }
        return transform;
    }

    [[nodiscard]] std::size_t Restore(SparseMap& trial, const SparseMap& reference, const SparseMap& moving) const {
        std::size_t bridges = 0;
        for (std::pair<const TrackId, Landmark>& entry : trial.landmarks) {
            for (const std::pair<const FrameId, Eigen::Vector2d>& observation : this->tracks_.at(entry.first)) {
                if (trial.cameras.contains(observation.first) &&
                    Reprojection(trial.cameras.at(observation.first), entry.second.position, observation.second) <= 3) {
                    entry.second.observations.emplace(observation);
                }
            }
            bool a = false, b = false;
            for (const std::pair<const FrameId, Eigen::Vector2d>& observation : entry.second.observations) {
                a = a || reference.cameras.contains(observation.first);
                b = b || moving.cameras.contains(observation.first);
            }
            bridges += a && b;
        }
        return bridges;
    }

    const std::vector<Observations>& tracks_;
    const SparseMap& accepted_;
    const std::function<void(const std::string&)>& progress_;
    std::map<FrameId, const DepthFrame*> frames_;
    std::size_t solves_ = 0;
};
} // namespace

void RecoverInitializationBridges(SparseMap& model, const std::vector<Observations>& tracks,
    const std::vector<DepthFrame>& frames, const SparseMap& accepted,
    const std::function<void(const std::string&)>& progress) {
    InitializationRecovery recovery(tracks, frames, accepted, progress);
    recovery.Run(model);
}
} // namespace stereoforge::optimization
