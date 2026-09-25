#include "stereoforge/optimization/map_builder.hpp"
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace stereoforge::optimization {
namespace {
double Median(std::vector<double> values) {
    if (values.empty()) { throw std::runtime_error("Cannot take an empty median"); }
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    return n % 2 ? values[n/2] : (values[n/2-1]+values[n/2])/2;
}
bool SamePixel(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    return std::nearbyint(a.x()*10000) == std::nearbyint(b.x()*10000) &&
           std::nearbyint(a.y()*10000) == std::nearbyint(b.y()*10000);
}
void CheckSupport(const SparseMap& model) {
    if (model.cameras.empty()) { throw std::runtime_error("Empty map"); }
    std::map<FrameId, std::size_t> support;
    std::map<FrameId, std::set<FrameId>> adjacency;
    for (const std::pair<const TrackId, Landmark>& entry : model.landmarks) {
        const Observations& obs = entry.second.observations;
        if (obs.size() < 3) { throw std::runtime_error("Landmark has fewer than three observations"); }
        const FrameId first = obs.begin()->first;
        for (const std::pair<const FrameId, Eigen::Vector2d>& o : obs) {
            ++support[o.first];
            adjacency[first].insert(o.first);
            adjacency[o.first].insert(first);
        }
    }
    std::set<FrameId> reached;
    std::vector<FrameId> pending{model.cameras.begin()->first};
    while (!pending.empty()) {
        const FrameId frame = pending.back(); pending.pop_back();
        if (!reached.insert(frame).second) { continue; }
        pending.insert(pending.end(), adjacency[frame].begin(), adjacency[frame].end());
    }
    for (const std::pair<const FrameId, Camera>& camera : model.cameras) {
        if (support[camera.first] < 6 || !reached.contains(camera.first)) {
            throw std::runtime_error("Map is disconnected or camera support is below six at frame " + std::to_string(camera.first));
        }
    }
}
}
double Reprojection(const Camera& camera, const Eigen::Vector3d& point, const Eigen::Vector2d& pixel) {
    const Eigen::Vector3d local = camera.rotation.transpose()*(point-camera.center);
    if (!local.allFinite() || local.z() <= 0) { return std::numeric_limits<double>::infinity(); }
    const Eigen::Vector3d projected = camera.intrinsics*local;
    return (projected.head<2>()/projected.z()-pixel).norm();
}
MapBuilder::MapBuilder(int device, int iterations, bool check_jacobians)
    : solver_(device), iterations_(iterations), check_jacobians_(check_jacobians) {
    if (iterations < 1 || iterations > 500) { throw std::invalid_argument("BA iterations must be 1..500"); }
}
void MapBuilder::SetTracks(std::vector<Observations> tracks) {
    if (this->accepted_windows_ != 0) { throw std::runtime_error("Cannot replace tracks in an active map"); }
    this->tracks_ = std::move(tracks);
    this->frame_tracks_.clear();
    std::set<std::tuple<FrameId, double, double>> measured;
    for (std::size_t i = 0; i < this->tracks_.size(); ++i) {
        for (const std::pair<const FrameId, Eigen::Vector2d>& observation : this->tracks_[i]) {
            if (!observation.second.allFinite()) { throw std::invalid_argument("Nonfinite track pixel"); }
            if (!measured.emplace(observation.first, std::nearbyint(observation.second.x()*10000),
                                  std::nearbyint(observation.second.y()*10000)).second) {
                throw std::invalid_argument("Ambiguous measured feature belongs to multiple global tracks");
            }
            this->frame_tracks_[observation.first].push_back(static_cast<TrackId>(i));
        }
    }
}
const SparseMap& MapBuilder::Map() const { return this->map_; }
std::size_t MapBuilder::AcceptedWindows() const { return this->accepted_windows_; }
SparseMap MapBuilder::Initialize(const std::vector<DepthFrame>& frames) const {
    SparseMap model;
    std::map<FrameId, const DepthFrame*> images;
    std::set<TrackId> candidates;
    for (const DepthFrame& frame : frames) {
        const std::size_t pixels = static_cast<std::size_t>(frame.camera.width)*frame.camera.height;
        if (frame.camera.width <= 0 || frame.camera.height <= 0 || frame.depth.size() != pixels || frame.rgb.size() != 3*pixels ||
            !frame.camera.rotation.allFinite() || !frame.camera.center.allFinite() || !frame.camera.intrinsics.allFinite() ||
            frame.camera.intrinsics(0,0) <= 0 || frame.camera.intrinsics(1,1) <= 0 ||
            !model.cameras.emplace(frame.id, frame.camera).second) { throw std::invalid_argument("Invalid VGGT frame"); }
        images.emplace(frame.id, &frame);
        if (this->frame_tracks_.contains(frame.id)) {
            const std::vector<TrackId>& tracks = this->frame_tracks_.at(frame.id);
            candidates.insert(tracks.begin(), tracks.end());
        }
    }
    for (const TrackId id : candidates) {
        Observations observations;
        std::array<std::vector<double>, 3> positions;
        for (const std::pair<const FrameId, Eigen::Vector2d>& observation : this->tracks_.at(id)) {
            if (!images.contains(observation.first)) { continue; }
            const DepthFrame& frame = *images.at(observation.first);
            const Camera& camera = frame.camera;
            const Eigen::Vector2d& uv = observation.second;
            observations.emplace(observation);
            if (uv.x() < 0 || uv.y() < 0 || uv.x() >= camera.width || uv.y() >= camera.height) { continue; }
            const int x = std::min(camera.width-1, static_cast<int>(std::nearbyint(uv.x())));
            const int y = std::min(camera.height-1, static_cast<int>(std::nearbyint(uv.y())));
            const double depth = frame.depth[y*camera.width+x];
            if (!std::isfinite(depth) || depth <= 0) { continue; }
            const Eigen::Vector3d ray = camera.intrinsics.inverse()*Eigen::Vector3d(uv.x(), uv.y(), 1);
            const Eigen::Vector3d xyz = camera.rotation*(ray*depth)+camera.center;
            for (int axis = 0; axis < 3; ++axis) { positions[axis].push_back(xyz[axis]); }
        }
        if (observations.size() < 3 || positions[0].empty()) { continue; }
        Landmark point;
        for (int axis = 0; axis < 3; ++axis) { point.position[axis] = Median(std::move(positions[axis])); }
        for (const std::pair<const FrameId, Eigen::Vector2d>& observation : observations) {
            if (Reprojection(model.cameras.at(observation.first), point.position, observation.second) <= 14) {
                point.observations.emplace(observation);
            }
        }
        if (point.observations.size() < 3) { continue; }
        const DepthFrame& color = *images.at(point.observations.begin()->first);
        const Eigen::Vector2d uv = point.observations.begin()->second;
        const int x = std::clamp(static_cast<int>(std::nearbyint(uv.x())), 0, color.camera.width-1);
        const int y = std::clamp(static_cast<int>(std::nearbyint(uv.y())), 0, color.camera.height-1);
        for (int c = 0; c < 3; ++c) { point.color[c] = color.rgb[3*(y*color.camera.width+x)+c]; }
        model.landmarks.emplace(id, std::move(point));
    }
    if (model.landmarks.size() < 60) { throw std::runtime_error("Window initialization has fewer than 60 landmarks"); }
    CheckSupport(model);
    return model;
}
void MapBuilder::Optimize(SparseMap& model, bool local) const {
    CheckSupport(model);
    const Eigen::Vector3d origin = model.cameras.begin()->second.center;
    Eigen::Vector3d previous = origin;
    std::vector<double> steps;
    std::vector<FrameId> camera_ids;
    std::map<FrameId, std::size_t> camera_index;
    for (const std::pair<const FrameId, Camera>& camera : model.cameras) {
        const double distance = (camera.second.center-previous).norm();
        if (distance > 1e-8) { steps.push_back(distance); }
        previous = camera.second.center;
        camera_index.emplace(camera.first, camera_ids.size()); camera_ids.push_back(camera.first);
    }
    if (steps.empty()) { throw std::runtime_error("Degenerate camera baseline"); }
    const double scale = Median(steps);
    BAInput input;
    input.options.use_gnc = local;
    input.options.huber_delta_pixels = local ? 0 : 3;
    input.options.lm_iterations = local ? 50 : this->iterations_;
    input.options.check_jacobians = this->check_jacobians_;

    for (const FrameId frame : camera_ids) {
        const Camera& camera = model.cameras.at(frame);
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose.topLeftCorner<3,3>() = camera.rotation.transpose();
        pose.topRightCorner<3,1>() = -camera.rotation.transpose()*((camera.center-origin)/scale);
        BACamera record;
        for (int r = 0; r < 4; ++r) { for (int c = 0; c < 4; ++c) { record.world_to_camera[4*r+c] = static_cast<float>(pose(r,c)); } }
        record.intrinsics = {static_cast<float>(camera.intrinsics(0,0)), static_cast<float>(camera.intrinsics(1,1)),
                            static_cast<float>(camera.intrinsics(0,2)), static_cast<float>(camera.intrinsics(1,2))};
        input.cameras.push_back(record);
    }
    std::vector<TrackId> point_ids;
    for (const std::pair<const TrackId, Landmark>& entry : model.landmarks) {
        const Eigen::Vector3d xyz = (entry.second.position-origin)/scale;
        const std::size_t index = point_ids.size(); point_ids.push_back(entry.first);
        input.points.push_back({static_cast<float>(xyz.x()), static_cast<float>(xyz.y()), static_cast<float>(xyz.z())});
        for (const std::pair<const FrameId, Eigen::Vector2d>& o : entry.second.observations) {
            input.observations.push_back({camera_index.at(o.first), index, {static_cast<float>(o.second.x()), static_cast<float>(o.second.y())}});
        }
    }
    const BAResult result = this->solver_.Solve(input);
    if (!result.report.optimization_complete) {
        throw std::runtime_error(local ? "Local BA did not converge" : "Joint BA exhausted its iteration budget; increase --lm-iterations");
    }
    for (std::size_t i = 0; i < camera_ids.size(); ++i) {
        Camera& camera = model.cameras.at(camera_ids[i]);
        Eigen::Matrix3d r; Eigen::Vector3d t;
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) { r(row,col) = result.cameras[i].world_to_camera[4*row+col]; }
            t[row] = result.cameras[i].world_to_camera[4*row+3];
        }
        camera.rotation = r.transpose(); camera.center = -r.transpose()*t*scale+origin;
        camera.intrinsics(0,0) = result.cameras[i].intrinsics[0]; camera.intrinsics(1,1) = result.cameras[i].intrinsics[1];
    }
    for (std::size_t i = 0; i < point_ids.size(); ++i) {
        Landmark& point = model.landmarks.at(point_ids[i]);
        point.position = Eigen::Vector3d(result.points[i][0], result.points[i][1], result.points[i][2])*scale+origin;
        std::erase_if(point.observations, [&](const std::pair<const FrameId, Eigen::Vector2d>& o) {
            return Reprojection(model.cameras.at(o.first), point.position, o.second) > 3;
        });
    }
    std::erase_if(model.landmarks, [](const std::pair<const TrackId, Landmark>& p) { return p.second.observations.size() < 3; });
    CheckSupport(model);
}
SparseMap MapBuilder::Combine(const SparseMap& local) const {
    SparseMap combined = this->map_;
    combined.cameras.insert(local.cameras.begin(), local.cameras.end());
    std::size_t joined = 0;
    for (const std::pair<const TrackId, Landmark>& entry : local.landmarks) {
        if (combined.landmarks.contains(entry.first)) {
            Landmark& point = combined.landmarks.at(entry.first);
            bool conflict = false;
            for (const std::pair<const FrameId, Eigen::Vector2d>& o : entry.second.observations) {
                if (point.observations.contains(o.first) && (point.observations.at(o.first)-o.second).norm() > 1e-3) { conflict = true; }
            }
            if (conflict) { continue; }
            std::size_t support = 0;
            for (const std::pair<const FrameId, Eigen::Vector2d>& o : entry.second.observations) {
                if (Reprojection(combined.cameras.at(o.first), point.position, o.second) <= 4) {
                    ++support; point.observations.emplace(o);
                }
            }
            joined += support >= 3;
        } else {
            Landmark point = entry.second;
            std::erase_if(point.observations, [&](const std::pair<const FrameId, Eigen::Vector2d>& o) {
                return Reprojection(combined.cameras.at(o.first), point.position, o.second) > 14;
            });
            if (point.observations.size() >= 3) { combined.landmarks.emplace(entry.first, std::move(point)); }
        }
    }
    if (joined < 20) { throw std::runtime_error("Fewer than 20 compatible shared tracks"); }
    // Do not reintroduce previous held-out measurements from the incoming window.
    for (const Boundary& boundary : this->boundaries_) {
        for (const Holdout& held : boundary.observations) {
            if (!combined.landmarks.contains(held.track)) { continue; }
            Observations& obs = combined.landmarks.at(held.track).observations;
            if (obs.contains(held.frame) && SamePixel(obs.at(held.frame), held.pixel)) { obs.erase(held.frame); }
        }
    }
    std::erase_if(combined.landmarks, [](const std::pair<const TrackId, Landmark>& p) { return p.second.observations.size() < 3; });
    CheckSupport(combined);
    return combined;
}
Boundary MapBuilder::Withhold(SparseMap& combined, const SparseMap& local) const {
    Boundary boundary;
    std::map<FrameId, std::size_t> support, counts;
    for (const std::pair<const FrameId, Camera>& camera : combined.cameras) { boundary.cameras.insert(camera.first); }
    for (const std::pair<const FrameId, Camera>& camera : local.cameras) {
        if (this->map_.cameras.contains(camera.first)) { boundary.shared.insert(camera.first); }
    }
    for (const std::pair<const TrackId, Landmark>& point : combined.landmarks) {
        for (const std::pair<const FrameId, Eigen::Vector2d>& obs : point.second.observations) { ++support[obs.first]; }
    }
    for (std::pair<const TrackId, Landmark>& entry : combined.landmarks) {
        if (!this->map_.landmarks.contains(entry.first) || !local.landmarks.contains(entry.first) || entry.second.observations.size() < 4) { continue; }
        // Same measured pixel in a shared frame, not identity alone, establishes a holdout track.
        bool matched = false;
        for (const std::pair<const FrameId, Eigen::Vector2d>& obs : this->map_.landmarks.at(entry.first).observations) {
            const Observations& other = local.landmarks.at(entry.first).observations;
            if (boundary.shared.contains(obs.first) && other.contains(obs.first) && SamePixel(obs.second, other.at(obs.first))) { matched = true; break; }
        }
        if (!matched) { continue; }
        FrameId selected = -1;
        for (const std::pair<const FrameId, Eigen::Vector2d>& obs : entry.second.observations) {
            if (boundary.shared.contains(obs.first) && support[obs.first] > 10 &&
                (selected == -1 || counts[obs.first] < counts[selected])) { selected = obs.first; }
        }
        if (selected == -1) { continue; }
        const Eigen::Vector2d pixel = entry.second.observations.at(selected);
        entry.second.observations.erase(selected);
        boundary.observations.push_back({entry.first, selected, pixel, entry.second.observations});
        --support[selected]; ++counts[selected];
    }
    if (boundary.observations.size() < 40) { throw std::runtime_error("Fewer than 40 held-out overlap observations"); }
    for (const FrameId frame : boundary.shared) {
        if (counts[frame] < 5) { throw std::runtime_error("Fewer than five held-out observations for frame " + std::to_string(frame)); }
    }
    CheckSupport(combined);
    return boundary;
}
void MapBuilder::Validate(const SparseMap& model, const Boundary& boundary) const {
    std::map<FrameId, std::size_t> counts, passed;
    std::size_t total = 0;
    for (const Holdout& held : boundary.observations) {
        ++counts[held.frame];
        if (!model.landmarks.contains(held.track) || !model.cameras.contains(held.frame)) { continue; }
        const Landmark& point = model.landmarks.at(held.track);
        std::size_t support = 0;
        for (const std::pair<const FrameId, Eigen::Vector2d>& obs : held.training) {
            if (point.observations.contains(obs.first) && SamePixel(obs.second, point.observations.at(obs.first))) { ++support; }
        }
        if (support >= 2 && Reprojection(model.cameras.at(held.frame), point.position, held.pixel) <= 5) { ++total; ++passed[held.frame]; }
    }
    if (total < .8*boundary.observations.size()) { throw std::runtime_error("Overlap validation below 80% overall"); }
    for (const FrameId frame : boundary.shared) {
        if (passed[frame] < .8*counts[frame]) { throw std::runtime_error("Overlap validation below 80% at frame " + std::to_string(frame)); }
    }
    for (const FrameId frame : boundary.cameras) {
        if (!model.cameras.contains(frame)) { throw std::runtime_error("Validation lost camera " + std::to_string(frame)); }
    }
}
void MapBuilder::AddWindow(const std::vector<DepthFrame>& frames, const std::function<void(const std::string&)>& progress) {
    progress("initializing landmarks");
    SparseMap local = this->Initialize(frames);
    progress("local cuNLS BA");
    this->Optimize(local, true);
    if (this->accepted_windows_ == 0) { this->map_ = std::move(local); ++this->accepted_windows_; return; }
    progress("shared-camera/landmark Sim(3)");
    SparseMap candidate;
    Boundary recovery_checks;
    try {
        const Similarity alignment = Align(this->map_, local);
        Transform(local, alignment);
        progress("combining tracks");
        candidate = this->Combine(local);
    } catch (const InsufficientAlignmentSupport& error) {
        progress(std::string("PnP recovery: ") + error.what());
        candidate = this->Recover(frames, local, recovery_checks, progress);
    }
    const Boundary boundary = this->Withhold(candidate, local);
    progress("joint cuNLS BA");
    this->Optimize(candidate, false);
    progress("validating overlaps");
    this->Validate(candidate, boundary);
    if (!recovery_checks.observations.empty()) { this->Validate(candidate, recovery_checks); }
    for (const Boundary& previous : this->boundaries_) { this->Validate(candidate, previous); }
    // Transactional commit: a rejected candidate never mutates the accepted map.
    this->map_ = std::move(candidate);
    this->boundaries_.push_back(boundary);
    if (!recovery_checks.observations.empty()) { this->boundaries_.push_back(std::move(recovery_checks)); }
    ++this->accepted_windows_;
}
} // namespace stereoforge::optimization
