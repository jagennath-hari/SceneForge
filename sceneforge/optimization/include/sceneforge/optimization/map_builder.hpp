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

#pragma once
#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include "sceneforge/optimization/rerun_recorder.hpp"
#include <set>
#include <string>
#include <stdexcept>
#include <vector>
#include "sceneforge/optimization/bundle_adjuster.hpp"

namespace sceneforge::optimization {
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
[[nodiscard]] Similarity Align(const SparseMap& reference, const SparseMap& local,
                               const std::vector<DepthFrame>& previous, const std::vector<DepthFrame>& incoming,
                               const std::function<void(const std::string&)>& progress);
void Transform(SparseMap& model, const Similarity& transform);
class MapBuilder final {
public:
    MapBuilder(int device, int iterations, bool check_jacobians);
    void RerunDense(const float* xyz, const std::uint8_t* rgb, std::size_t count);
    void RerunEvent(const std::string& message);
    void RerunKeyframe(const DepthFrame& frame);
    void PreviewWindow(const std::vector<DepthFrame>& frames);
    void EnableRerun(const std::string& path);
    void SetTracks(std::vector<Observations> tracks);
    void AddWindow(const std::vector<DepthFrame>& frames, const std::vector<DepthFrame>& reference_frames,
                   const std::function<void(const std::string&)>& progress);
    [[nodiscard]] std::vector<std::size_t> RankWindows(const std::vector<std::vector<FrameId>>& windows) const;
    void Finalize(const std::function<void(const std::string&)>& progress);
    [[nodiscard]] const SparseMap& Map() const;
    [[nodiscard]] std::size_t AcceptedWindows() const;
    [[nodiscard]] const std::vector<FrameId>& UnanchoredFrames() const { return this->unanchored_frames_; }
    [[nodiscard]] bool SharedCalibrationComplete() const { return this->shared_calibration_complete_; }
private:
    [[nodiscard]] SparseMap Initialize(const std::vector<DepthFrame>& frames,
        const std::function<void(const std::string&)>& progress) const;
    void Optimize(SparseMap& model, bool local, bool shared_calibration = false,
                  const std::function<void(const std::string&)>* progress = nullptr) const;
    [[nodiscard]] SparseMap Combine(const SparseMap& local) const;
    [[nodiscard]] Boundary Withhold(SparseMap& combined, const SparseMap& local,
                                    const std::function<void(const std::string&)>& progress) const;
    void Validate(const SparseMap& model, const Boundary& boundary,
                  const std::function<void(const std::string&)>* global_progress = nullptr) const;
    void Record(const SparseMap& model, std::uint32_t stage,
                const std::function<void(const std::string&)>& progress, const DepthFrame* image = nullptr);
    std::unique_ptr<RerunRecorder> recorder_;
    BundleAdjuster solver_;
    int iterations_;
    bool shared_calibration_complete_ = false;
    bool check_jacobians_;
    std::vector<FrameId> unanchored_frames_;
    std::vector<Observations> tracks_;
    std::map<FrameId, std::vector<TrackId>> frame_tracks_;
    SparseMap map_;
    std::vector<Boundary> boundaries_;
    std::size_t accepted_windows_ = 0;
};
} // namespace sceneforge::optimization
