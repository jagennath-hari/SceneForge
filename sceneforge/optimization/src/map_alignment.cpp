#include "sceneforge/optimization/map_builder.hpp"
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace sceneforge::optimization {
namespace {
constexpr double kPi = 3.14159265358979323846;
double Median(std::vector<double> values) {
    if (values.empty()) { throw std::runtime_error("No scale evidence in shared cameras or depths"); }
    std::sort(values.begin(),values.end());
    return values.size()%2 ? values[values.size()/2] : (values[values.size()/2-1]+values[values.size()/2])/2;
}
Eigen::Vector3d MedianVector(const std::vector<Eigen::Vector3d>& values) {
    Eigen::Vector3d result;
    for (int axis = 0; axis < 3; ++axis) {
        std::vector<double> coordinates;
        for (const Eigen::Vector3d& value : values) { coordinates.push_back(value[axis]); }
        result[axis] = Median(coordinates);
    }
    return result;
}
double RotationError(const Eigen::Matrix3d& a, const Eigen::Matrix3d& b) {
    return std::acos(std::clamp(((a.transpose()*b).trace()-1)/2,-1.0,1.0));
}
Eigen::Matrix3d MeanRotation(const std::vector<Eigen::Matrix3d>& rotations) {
    // Start at the angular medoid, then use Huber-weighted chordal averaging.
    Eigen::Matrix3d result = rotations.front();
    double best = std::numeric_limits<double>::infinity();
    for (const Eigen::Matrix3d& candidate : rotations) {
        double cost = 0;
        for (const Eigen::Matrix3d& rotation : rotations) { cost += RotationError(candidate,rotation); }
        if (cost < best) { best = cost; result = candidate; }
    }
    for (int iteration = 0; iteration < 10; ++iteration) {
        Eigen::Matrix3d sum = Eigen::Matrix3d::Zero();
        for (const Eigen::Matrix3d& rotation : rotations) {
            const double angle = RotationError(result,rotation);
            sum += std::min(1.0,(5*kPi/180)/std::max(angle,1e-12))*rotation;
        }
        const Eigen::JacobiSVD<Eigen::Matrix3d> svd(sum,Eigen::ComputeFullU|Eigen::ComputeFullV);
        Eigen::Vector3d signs(1,1,(svd.matrixU()*svd.matrixV().transpose()).determinant() < 0 ? -1 : 1);
        const Eigen::Matrix3d next = svd.matrixU()*signs.asDiagonal()*svd.matrixV().transpose();
        const double change = RotationError(result,next);
        result = next;
        if (change < 1e-8) { break; }
    }
    return result;
}
struct DepthGauge {
    double factor;
    double scene_depth;
};
std::optional<DepthGauge> Gauge(const SparseMap& model, const DepthFrame& frame) {
    const Camera& camera = model.cameras.at(frame.id);
    std::vector<double> ratios, depths;
    for (const std::pair<const TrackId,Landmark>& entry : model.landmarks) {
        const Landmark& point = entry.second;
        if (!point.observations.contains(frame.id)) { continue; }
        const Eigen::Vector2d pixel = point.observations.at(frame.id);
        if (!pixel.allFinite() || Reprojection(camera,point.position,pixel) > 3) { continue; }
        const int x = static_cast<int>(std::lround(pixel.x())), y = static_cast<int>(std::lround(pixel.y()));
        if (x < 0 || y < 0 || x >= frame.camera.width || y >= frame.camera.height) { continue; }
        const double raw = frame.depth[static_cast<std::size_t>(y)*frame.camera.width+x];
        const double optimized = (camera.rotation.transpose()*(point.position-camera.center)).z();
        if (std::isfinite(raw) && raw > 0 && std::isfinite(optimized) && optimized > 0) {
            ratios.push_back(std::log(optimized/raw)); depths.push_back(optimized);
        }
    }
    // This calibrates an approximate depth prior to the current BA gauge, not
    // a dense-depth correction. No shared track IDs or triangulation angle gate.
    if (ratios.size() < 3) { return std::nullopt; }
    return DepthGauge{std::exp(Median(ratios)),Median(depths)};
}
void CheckImage(const DepthFrame& frame) {
    if (frame.camera.width <= 0 || frame.camera.height <= 0 ||
        frame.depth.size() != static_cast<std::size_t>(frame.camera.width)*frame.camera.height ||
        frame.rgb.size() != 3*frame.depth.size()) {
        throw std::runtime_error("Invalid shared depth/image dimensions");
    }
}
}
Similarity Align(const SparseMap& reference, const SparseMap& local,
                 const std::vector<DepthFrame>& previous, const std::vector<DepthFrame>& incoming,
                 const std::function<void(const std::string&)>& progress) {
    std::vector<FrameId> shared;
    std::vector<Eigen::Matrix3d> rotations;
    for (const std::pair<const FrameId,Camera>& entry : local.cameras) {
        if (!reference.cameras.contains(entry.first)) { continue; }
        shared.push_back(entry.first);
        rotations.push_back(reference.cameras.at(entry.first).rotation*entry.second.rotation.transpose());
    }
    if (shared.empty()) { throw std::runtime_error("Camera alignment needs a shared frame"); }
    Similarity result;
    result.rotation = MeanRotation(rotations);
    std::map<FrameId,const DepthFrame*> old_frames, new_frames;
    for (const DepthFrame& frame : previous) { CheckImage(frame); old_frames.emplace(frame.id,&frame); }
    for (const DepthFrame& frame : incoming) { CheckImage(frame); new_frames.emplace(frame.id,&frame); }
    std::vector<double> depth_scales, reference_depths, local_depths;
    for (const FrameId id : shared) {
        if (!old_frames.contains(id) || !new_frames.contains(id)) { continue; }
        const DepthFrame& a = *old_frames.at(id);
        const DepthFrame& b = *new_frames.at(id);
        if (a.camera.width != b.camera.width || a.camera.height != b.camera.height || a.rgb != b.rgb) {
            throw std::runtime_error("Shared processed RGB differs at frame " + std::to_string(id));
        }
        const std::optional<DepthGauge> ga = Gauge(reference,a), gb = Gauge(local,b);
        if (!ga || !gb) { continue; }
        reference_depths.push_back(ga->scene_depth); local_depths.push_back(gb->scene_depth);
        std::vector<double> ratios;
        // Equal frame weighting and a bounded deterministic pixel sample.
        const std::size_t stride = std::max<std::size_t>(1,a.depth.size()/2048);
        for (std::size_t pixel = 0; pixel < a.depth.size(); pixel += stride) {
            const double da = a.depth[pixel]*ga->factor, db = b.depth[pixel]*gb->factor;
            if (std::isfinite(da) && std::isfinite(db) && da > 0 && db > 0) { ratios.push_back(std::log(da/db)); }
        }
        if (!ratios.empty()) { depth_scales.push_back(Median(ratios)); }
    }
    // Use center displacement only when it is appreciable relative to scene
    // depth in BOTH gauges. Ratios of near-zero baselines are not scale evidence.
    std::vector<double> center_scales;
    if (!reference_depths.empty() && !local_depths.empty()) {
        const double target_floor = .02*Median(reference_depths), source_floor = .02*Median(local_depths);
        for (std::size_t i = 0; i < shared.size(); ++i) {
            for (std::size_t j = 0; j < i; ++j) {
                const Eigen::Vector3d source = result.rotation*(local.cameras.at(shared[i]).center-local.cameras.at(shared[j]).center);
                const Eigen::Vector3d target = reference.cameras.at(shared[i]).center-reference.cameras.at(shared[j]).center;
                if (source.norm() <= source_floor || target.norm() <= target_floor) { continue; }
                if (source.dot(target)/(source.norm()*target.norm()) < .8) { continue; }
                const double scale = source.dot(target)/source.squaredNorm();
                if (std::isfinite(scale) && scale > 0) { center_scales.push_back(std::log(scale)); }
            }
        }
    }
    const bool from_centers = center_scales.size() >= 3;
    result.scale = std::exp(Median(from_centers ? center_scales : depth_scales));
    std::vector<Eigen::Vector3d> translations;
    for (const FrameId id : shared) {
        translations.push_back(reference.cameras.at(id).center-result.scale*result.rotation*local.cameras.at(id).center);
    }
    result.translation = MedianVector(translations);
    if (!std::isfinite(result.scale) || result.scale <= 0 || !result.rotation.allFinite() || !result.translation.allFinite()) {
        throw std::runtime_error("Shared-camera alignment produced invalid Sim(3)");
    }
    std::vector<double> angles;
    for (const Eigen::Matrix3d& rotation : rotations) { angles.push_back(RotationError(result.rotation,rotation)*180/kPi); }
    if (Median(angles) > 10) {
        progress("WARNING: Shared camera rotation disagreement: median " + std::to_string(Median(angles)) + " degrees; joint BA must refine the initialization");
    }
    progress(from_centers ? "camera alignment: scale from shared centers" : "camera alignment: scale from shared-frame depth prior");
    return result;
}
void Transform(SparseMap& model, const Similarity& transform) {
    for (std::pair<const FrameId,Camera>& entry : model.cameras) {
        entry.second.rotation = transform.rotation*entry.second.rotation;
        entry.second.center = transform.scale*transform.rotation*entry.second.center+transform.translation;
    }
    for (std::pair<const TrackId,Landmark>& entry : model.landmarks) {
        entry.second.position = transform.scale*transform.rotation*entry.second.position+transform.translation;
    }
}
} // namespace sceneforge::optimization
