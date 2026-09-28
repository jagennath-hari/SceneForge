#include "stereoforge/optimization/map_builder.hpp"
#include "stereoforge/optimization/map_bridge.hpp"
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
using SupportCounts = std::map<FrameId, std::size_t>;
SupportCounts CountSupport(const SparseMap& model) {
    SupportCounts counts;
    for (const std::pair<const TrackId, Landmark>& point : model.landmarks) {
        for (const std::pair<const FrameId, Eigen::Vector2d>& observation : point.second.observations) {
            ++counts[observation.first];
        }
    }
    return counts;
}
std::size_t SupportAt(const SupportCounts& counts, FrameId frame) {
    return counts.contains(frame) ? counts.at(frame) : 0;
}
void CheckSupport(const SparseMap& model, const std::string& stage,
                  const SupportCounts* before = nullptr, const SupportCounts* after_pixel_filter = nullptr,
                  bool require_connected = true) {
    if (model.cameras.empty()) { throw std::runtime_error(stage + ": empty map"); }
    std::map<FrameId, std::size_t> support;
    std::map<FrameId, std::set<FrameId>> adjacency;
    for (const std::pair<const TrackId, Landmark>& entry : model.landmarks) {
        const Observations& obs = entry.second.observations;
        if (obs.size() < 3) { throw std::runtime_error(stage + ": landmark has fewer than three observations"); }
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
        if (support[camera.first] < 6 || (require_connected && !reached.contains(camera.first))) {
            std::string message = stage + ": frame " + std::to_string(camera.first) +
                " has " + std::to_string(support[camera.first]) + " surviving landmark observations (minimum 6); " +
                "connected to anchor " + std::to_string(model.cameras.begin()->first) + "=" +
                (reached.contains(camera.first) ? "yes" : "no") + "; anchor component " +
                std::to_string(reached.size()) + "/" + std::to_string(model.cameras.size()) + " cameras";
            if (before != nullptr) {
                message += "; observations before this stage=" + std::to_string(SupportAt(*before,camera.first));
            }
            if (after_pixel_filter != nullptr) {
                message += "; after positive-depth filtering=" + std::to_string(SupportAt(*after_pixel_filter,camera.first)) +
                    "; after removing tracks with fewer than 3 views=" + std::to_string(support[camera.first]);
            }
            throw std::runtime_error(message);
        }
    }
}
} // namespace

std::vector<SparseMap> ConnectedGroups(const SparseMap& model) {
    std::map<FrameId, std::vector<FrameId>> adjacency;
    for (const std::pair<const TrackId, Landmark>& point : model.landmarks) {
        const FrameId first = point.second.observations.begin()->first;
        for (const std::pair<const FrameId, Eigen::Vector2d>& observation : point.second.observations) {
            adjacency[first].push_back(observation.first);
            adjacency[observation.first].push_back(first);
        }
    }
    std::map<FrameId, std::size_t> membership;
    std::vector<SparseMap> groups;
    for (const std::pair<const FrameId, Camera>& camera : model.cameras) {
        if (membership.contains(camera.first)) { continue; }
        const std::size_t index = groups.size();
        groups.emplace_back();
        std::vector<FrameId> pending{camera.first};
        while (!pending.empty()) {
            const FrameId frame = pending.back(); pending.pop_back();
            if (!membership.emplace(frame,index).second) { continue; }
            groups.back().cameras.emplace(frame,model.cameras.at(frame));
            const std::vector<FrameId>& neighbors = adjacency[frame];
            pending.insert(pending.end(),neighbors.begin(),neighbors.end());
        }
    }
    for (const std::pair<const TrackId, Landmark>& point : model.landmarks) {
        groups.at(membership.at(point.second.observations.begin()->first)).landmarks.emplace(point);
    }
    return groups;
}

double Reprojection(const Camera& camera, const Eigen::Vector3d& point, const Eigen::Vector2d& pixel) {
    const Eigen::Vector3d local = camera.rotation.transpose()*(point-camera.center);
    if (!local.allFinite() || local.z() <= 0) { return std::numeric_limits<double>::infinity(); }
    const Eigen::Vector3d projected = camera.intrinsics*local;
    return (projected.head<2>()/projected.z()-pixel).norm();
}
MapBuilder::MapBuilder(int device, int iterations, bool check_jacobians)
    : solver_(device), iterations_(iterations), check_jacobians_(check_jacobians) {
    if (iterations < 1 || iterations > 1000) { throw std::invalid_argument("BA iterations must be 1..1000"); }
}
void MapBuilder::RerunDense(const float* xyz, const std::uint8_t* rgb, std::size_t count) {
    if (!this->recorder_) { return; }
    try { this->recorder_->Dense(xyz,rgb,count); }
    catch (...) { this->recorder_.reset(); throw; }
}
void MapBuilder::RerunEvent(const std::string& message) {
    if (!this->recorder_) { return; }
    try { this->recorder_->Event(message); }
    catch (...) { this->recorder_.reset(); throw; }
}
void MapBuilder::RerunKeyframe(const DepthFrame& frame) {
    if (!this->recorder_) { return; }
    try { this->recorder_->Image(frame); }
    catch (...) { this->recorder_.reset(); throw; }
}
void MapBuilder::PreviewWindow(const std::vector<DepthFrame>& frames) {
    if (!this->recorder_ || frames.empty()) { return; }
    try {
        SparseMap preview;
        TrackId id = 0;
        for (const DepthFrame& frame : frames) {
            if (frame.camera.width <= 0 || frame.camera.height <= 0 ||
                frame.depth.size() != static_cast<std::size_t>(frame.camera.width)*frame.camera.height ||
                frame.rgb.size() != 3*frame.depth.size() || !frame.camera.intrinsics.allFinite() ||
                frame.camera.intrinsics(0,0) <= 0 || frame.camera.intrinsics(1,1) <= 0) {
                throw std::runtime_error("Invalid VGGT preview frame dimensions/calibration");
            }
            preview.cameras.emplace(frame.id,frame.camera);
            const std::size_t stride = std::max<std::size_t>(1,(frame.depth.size()+1999)/2000);
            for (std::size_t pixel=0; pixel<frame.depth.size(); pixel+=stride) {
                const double z = frame.depth[pixel];
                if (!std::isfinite(z) || z <= 0) { continue; }
                const double u = static_cast<double>(pixel % frame.camera.width);
                const double v = static_cast<double>(pixel / frame.camera.width);
                const Eigen::Vector3d ray((u-frame.camera.intrinsics(0,2))/frame.camera.intrinsics(0,0),
                                           (v-frame.camera.intrinsics(1,2))/frame.camera.intrinsics(1,1),1);
                Landmark point;
                point.position = frame.camera.rotation*(ray*z)+frame.camera.center;
                point.color = {frame.rgb[3*pixel],frame.rgb[3*pixel+1],frame.rgb[3*pixel+2]};
                preview.landmarks.emplace(id++,std::move(point));
            }
        }
        this->recorder_->Snapshot(preview,0);
        this->recorder_->Image(frames.front());
    } catch (...) { this->recorder_.reset(); throw; }
}
void MapBuilder::EnableRerun(const std::string& path) {
    this->recorder_ = std::make_unique<RerunRecorder>(path);
}
void MapBuilder::Record(const SparseMap& model, std::uint32_t stage,
                        const std::function<void(const std::string&)>& progress, const DepthFrame* image) {
    if (!this->recorder_) { return; }
    try {
        this->recorder_->Snapshot(model,stage);
        if (image != nullptr) { this->recorder_->Image(*image); }
    }
    catch (const std::exception& error) {
        this->recorder_.reset();
        progress(std::string("WARNING: Rerun recording disabled; reconstruction continues: ") + error.what());
    }
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
SparseMap MapBuilder::Initialize(const std::vector<DepthFrame>& frames,
    const std::function<void(const std::string&)>& progress) {
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
            if (std::isfinite(Reprojection(model.cameras.at(observation.first), point.position, observation.second))) {
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
    RecoverInitializationBridges(model, this->tracks_, frames, this->map_, progress);
    // Propose a smaller candidate only for isolated unsupported cameras. This
    // is evidence for the scheduler, not an accepted or partially committed map.
    // The candidate must be initialized again because removing a camera changes
    // depth medians, observation counts and the surviving track graph.
    const SupportCounts support = CountSupport(model);
    std::vector<FrameId> supported;
    for (const std::pair<const FrameId, Camera>& camera : model.cameras) {
        if (SupportAt(support, camera.first) >= 6) { supported.push_back(camera.first); }
    }
    const std::size_t excluded = model.cameras.size()-supported.size();
    const std::size_t maximum_excluded = std::min<std::size_t>(8, model.cameras.size()/4);
    if (excluded > 0 && excluded <= maximum_excluded && supported.size() >= 8) {
        this->supported_initialization_frames_ = std::move(supported);
    }
    CheckSupport(model, "VGGT landmark initialization", nullptr, nullptr, false);
    return model;
}
void MapBuilder::Optimize(SparseMap& model, bool local, bool shared_calibration,
                          const std::function<void(const std::string&)>* progress) const {
    const std::string stage = local ? "local BA" : "joint BA";
    CheckSupport(model, stage + " input");
    const SupportCounts before = CountSupport(model);
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
    input.options.shared_intrinsics = shared_calibration;
    input.options.optimize_principal = shared_calibration;
    // One smooth robust objective at every scale; no GNC pruning/retry branch.
    input.options.use_gnc = false;
    input.options.huber_delta_pixels = 3;
    input.options.lm_iterations = this->iterations_;
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
        // A capped solve is usable without claiming convergence. Solve()
        // already rejects nonfinite parameters and increasing objective values;
        // output support and geometry checks below still apply.
        if (result.report.lm_budget_exhausted) {
            if (progress != nullptr) {
                (*progress)(std::string("WARNING: ") + (shared_calibration ? "Final global BA" : (local ? "Local BA" : "Joint BA")) +
                    " reached its iteration limit; continuing to geometry validation (convergence not established)");
            }
        } else {
            throw std::runtime_error(stage + " did not produce a completed or budget-limited result");
        }
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
        camera.intrinsics(0,2) = result.cameras[i].intrinsics[2]; camera.intrinsics(1,2) = result.cameras[i].intrinsics[3];
    }
    for (std::size_t i = 0; i < point_ids.size(); ++i) {
        Landmark& point = model.landmarks.at(point_ids[i]);
        point.position = Eigen::Vector3d(result.points[i][0], result.points[i][1], result.points[i][2])*scale+origin;
        std::erase_if(point.observations, [&](const std::pair<const FrameId, Eigen::Vector2d>& o) {
            // Keep finite residuals in the robust objective, even outside its quadratic region.
            return !std::isfinite(Reprojection(model.cameras.at(o.first), point.position, o.second));
        });
    }
    const SupportCounts after_pixel_filter = CountSupport(model);
    std::erase_if(model.landmarks, [](const std::pair<const TrackId, Landmark>& p) { return p.second.observations.size() < 3; });
    CheckSupport(model, stage + " output", &before, &after_pixel_filter, !local);
}
SparseMap MapBuilder::Combine(const SparseMap& local) const {
    SparseMap combined = this->map_;
    combined.cameras.insert(local.cameras.begin(), local.cameras.end());
    for (const std::pair<const TrackId, Landmark>& entry : local.landmarks) {
        if (combined.landmarks.contains(entry.first)) {
            Landmark& point = combined.landmarks.at(entry.first);
            bool conflict = false;
            for (const std::pair<const FrameId, Eigen::Vector2d>& o : entry.second.observations) {
                if (point.observations.contains(o.first) && (point.observations.at(o.first)-o.second).norm() > 1e-3) { conflict = true; }
            }
            if (conflict) { continue; }
            for (const std::pair<const FrameId, Eigen::Vector2d>& o : entry.second.observations) {
                if (std::isfinite(Reprojection(combined.cameras.at(o.first), point.position, o.second))) {
                    point.observations.emplace(o);
                }
            }
        } else {
            Landmark point = entry.second;
            std::erase_if(point.observations, [&](const std::pair<const FrameId, Eigen::Vector2d>& o) {
                return !std::isfinite(Reprojection(combined.cameras.at(o.first), point.position, o.second));
            });
            if (point.observations.size() >= 3) { combined.landmarks.emplace(entry.first, std::move(point)); }
        }
    }
    std::erase_if(combined.landmarks, [](const std::pair<const TrackId, Landmark>& p) { return p.second.observations.size() < 3; });
    const SupportCounts established = CountSupport(this->map_);
    CheckSupport(combined, "combining tracks (before=accepted map)", &established);
    return combined;
}
void MapBuilder::AddWindow(const std::vector<DepthFrame>& frames, const std::vector<DepthFrame>& reference_frames,
                           const std::function<void(const std::string&)>& progress) {
    this->unanchored_frames_.clear();
    this->supported_initialization_frames_.clear();
    progress("initializing landmarks");
    const SparseMap initialized = this->Initialize(frames, progress);
    this->Record(initialized,0,progress,frames.empty() ? nullptr : &frames.front());
    std::vector<SparseMap> initial_groups = ConnectedGroups(initialized);
    if (this->accepted_windows_ == 0 && initial_groups.size() != 1) {
        throw std::runtime_error("Seed window is disconnected; no established map exists to anchor its groups");
    }
    SparseMap refined;
    // Each disconnected input component has its own gauge until Sim(3) alignment.
    for (std::size_t i = 0; i < initial_groups.size(); ++i) {
        progress("local cuNLS BA: group " + std::to_string(i+1) + "/" + std::to_string(initial_groups.size()));
        this->Optimize(initial_groups[i], true, false, &progress);
        refined.cameras.merge(initial_groups[i].cameras);
        refined.landmarks.merge(initial_groups[i].landmarks);
    }
    if (this->accepted_windows_ == 0) {
        CheckSupport(refined, "seed after local BA");
        this->map_ = std::move(refined);
        ++this->accepted_windows_;
        this->Record(this->map_,2,progress,frames.empty() ? nullptr : &frames.front());
        return;
    }
    std::vector<SparseMap> groups = ConnectedGroups(refined);
    SparseMap local;
    for (std::size_t i = 0; i < groups.size(); ++i) {
        SparseMap& group = groups[i];
        std::size_t shared = 0;
        for (const std::pair<const FrameId, Camera>& camera : group.cameras) {
            shared += this->map_.cameras.contains(camera.first);
        }
        const std::string label = "local group " + std::to_string(i+1) + "/" + std::to_string(groups.size()) +
            " (frames " + std::to_string(group.cameras.begin()->first) + ".." +
            std::to_string(group.cameras.rbegin()->first) + ", " + std::to_string(shared) + " shared cameras)";
        if (shared == 0) {
            for (const std::pair<const FrameId, Camera>& camera : group.cameras) {
                this->unanchored_frames_.push_back(camera.first);
            }
            throw std::runtime_error(label + ": cannot anchor to the established map");
        }
        progress("shared-camera Sim(3): " + label);
        try {
            const Similarity alignment = Align(this->map_, group, reference_frames, frames, progress);
            Transform(group, alignment);
        } catch (const std::runtime_error& error) {
            throw std::runtime_error(label + ": " + error.what());
        }
        // Components partition cameras and global tracks. All transforms are
        // estimated against the unchanged accepted map, never another new group.
        local.cameras.merge(group.cameras);
        local.landmarks.merge(group.landmarks);
    }
    this->Record(local,1,progress);
    progress("combining tracks");
    SparseMap candidate = this->Combine(local);
    progress("joint cuNLS BA");
    this->Optimize(candidate, false, false, &progress);
    // Solve checks the robust objective, finite geometry and graph support.
    // Commit transactionally; failed solves leave the accepted map unchanged.
    this->map_ = std::move(candidate);
    ++this->accepted_windows_;
    this->Record(this->map_,2,progress,frames.empty() ? nullptr : &frames.front());
}
std::vector<std::size_t> MapBuilder::RankWindows(const std::vector<std::vector<FrameId>>& windows) const {
    // Rank cheap graph evidence before loading VGGT tensors or running local BA.
    // Feature tracks alone cannot provide the shared-camera Sim(3) anchor.
    std::vector<std::tuple<std::size_t,std::size_t,std::size_t>> ranked;
    for (std::size_t index = 0; index < windows.size(); ++index) {
        std::map<TrackId,std::size_t> support;
        std::size_t cameras = 0;
        for (const FrameId frame : windows[index]) {
            cameras += this->map_.cameras.contains(frame);
            if (!this->frame_tracks_.contains(frame)) { continue; }
            for (const TrackId track : this->frame_tracks_.at(frame)) { ++support[track]; }
        }
        if (this->accepted_windows_ != 0 && cameras == 0) { continue; }
        std::size_t tracks = 0;
        for (const std::pair<const TrackId,std::size_t>& entry : support) {
            if (entry.second >= 3 && (this->accepted_windows_ == 0 || this->map_.landmarks.contains(entry.first))) { ++tracks; }
        }
        ranked.emplace_back(cameras,tracks,index);
    }
    std::sort(ranked.begin(),ranked.end(),[](const std::tuple<std::size_t,std::size_t,std::size_t>& a,
                                           const std::tuple<std::size_t,std::size_t,std::size_t>& b) {
        if (std::get<0>(a) != std::get<0>(b)) { return std::get<0>(a) > std::get<0>(b); }
        if (std::get<1>(a) != std::get<1>(b)) { return std::get<1>(a) > std::get<1>(b); }
        return std::get<2>(a) < std::get<2>(b);
    });
    std::vector<std::size_t> result;
    for (const std::tuple<std::size_t,std::size_t,std::size_t>& entry : ranked) { result.push_back(std::get<2>(entry)); }
    return result;
}
void MapBuilder::Finalize(const std::function<void(const std::string&)>& progress) {
    if (this->accepted_windows_ == 0) { throw std::runtime_error("No accepted map for global BA"); }
    SparseMap candidate = this->map_;
    const Camera& first = candidate.cameras.begin()->second;
    std::array<std::vector<double>,4> values;
    for (const std::pair<const FrameId,Camera>& entry : candidate.cameras) {
        if (entry.second.width != first.width || entry.second.height != first.height) {
            throw std::runtime_error("Shared calibration requires identical processed image dimensions");
        }
        values[0].push_back(entry.second.intrinsics(0,0)); values[1].push_back(entry.second.intrinsics(1,1));
        values[2].push_back(entry.second.intrinsics(0,2)); values[3].push_back(entry.second.intrinsics(1,2));
    }
    Eigen::Matrix3d intrinsics = first.intrinsics;
    intrinsics(0,0) = Median(values[0]); intrinsics(1,1) = Median(values[1]);
    intrinsics(0,2) = Median(values[2]); intrinsics(1,2) = Median(values[3]);
    for (std::pair<const FrameId,Camera>& entry : candidate.cameras) { entry.second.intrinsics = intrinsics; }
    progress("global BA: joint shared fx/fy/cx/cy, poses and landmarks");
    if (this->recorder_) { this->RerunEvent("Global BA: optimizing shared intrinsics, poses and landmarks; frustums update after optimization"); }
    this->Optimize(candidate,false,true,&progress);
    // The same objective and numerical/support safeguards apply to global BA.
    this->map_ = std::move(candidate);
    this->shared_calibration_complete_ = true;
    this->Record(this->map_,3,progress);

}
} // namespace stereoforge::optimization
