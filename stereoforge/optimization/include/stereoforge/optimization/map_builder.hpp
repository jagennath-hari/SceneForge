#pragma once
#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "stereoforge/optimization/bundle_adjuster.hpp"

namespace stereoforge::optimization {
using FrameId = std::int64_t;
using TrackId = std::int64_t;
using Observations = std::map<FrameId, Eigen::Vector2d>;
struct Camera {
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity(); // camera to world
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    Eigen::Matrix3d intrinsics = Eigen::Matrix3d::Identity();
    int height = 0;
    int width = 0;
};
struct Landmark {
    Eigen::Vector3d position;
    std::array<std::uint8_t, 3> color;
    Observations observations;
};
struct SparseMap {
    std::map<FrameId, Camera> cameras;
    std::map<TrackId, Landmark> landmarks;
};
struct DepthFrame {
    FrameId id;
    Camera camera;
    std::vector<float> depth;
    std::vector<std::uint8_t> rgb;
};
struct Similarity {
    double scale = 1;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
};
struct Holdout {
    TrackId track;
    FrameId frame;
    Eigen::Vector2d pixel;
    Observations training;
};
struct Boundary {
    std::vector<Holdout> observations;
    std::set<FrameId> shared;
    std::set<FrameId> cameras;
};
[[nodiscard]] double Reprojection(const Camera& camera, const Eigen::Vector3d& point, const Eigen::Vector2d& pixel);
[[nodiscard]] Similarity Align(const SparseMap& reference, const SparseMap& local);
void Transform(SparseMap& model, const Similarity& transform);
class MapBuilder final {
public:
    MapBuilder(int device, int iterations, bool check_jacobians);
    void SetTracks(std::vector<Observations> tracks);
    void AddWindow(const std::vector<DepthFrame>& frames, const std::function<void(const std::string&)>& progress);
    [[nodiscard]] const SparseMap& Map() const;
    [[nodiscard]] std::size_t AcceptedWindows() const;
private:
    [[nodiscard]] SparseMap Initialize(const std::vector<DepthFrame>& frames) const;
    void Optimize(SparseMap& model, bool local) const;
    [[nodiscard]] SparseMap Combine(const SparseMap& local) const;
    [[nodiscard]] Boundary Withhold(SparseMap& combined, const SparseMap& local) const;
    void Validate(const SparseMap& model, const Boundary& boundary) const;
    BundleAdjuster solver_;
    int iterations_;
    bool check_jacobians_;
    std::vector<Observations> tracks_;
    std::map<FrameId, std::vector<TrackId>> frame_tracks_;
    SparseMap map_;
    std::vector<Boundary> boundaries_;
    std::size_t accepted_windows_ = 0;
};
} // namespace stereoforge::optimization
