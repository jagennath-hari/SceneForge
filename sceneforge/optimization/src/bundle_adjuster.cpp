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

#include "sceneforge/optimization/bundle_adjuster.hpp"
#include "sceneforge/optimization/factors.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cunls/common/cublas_helper.h>
#include <cunls/common/cuda_stream.h>
#include <cunls/common/device_vector.h>
#include <cunls/common/helper.h>
#include <cunls/minimizer/levenberg_marquardt_minimizer.h>
#include <cunls/minimizer/problem.h>
#include <cunls/state/se3_state_batch.h>
#include <cunls/state/vector_state_batch.h>

namespace sceneforge::optimization {
// cuNLS error macros expand an unqualified LogError at the call site.
using cunls::LogError;

namespace {
void ValidatePose(const std::vector<float>& pose) {
    if (pose.size() != 16 || !std::all_of(pose.begin(), pose.end(), [](float v) { return std::isfinite(v); })) {
        throw std::invalid_argument("Invalid or nonfinite pose");
    }
    for (int i = 0; i < 4; ++i) {
        if (std::abs(pose[12+i] - (i == 3 ? 1.0f : 0.0f)) > 1e-5f) {
            throw std::invalid_argument("Pose must be a homogeneous world-to-camera matrix");
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double dot = 0;
            for (int k = 0; k < 3; ++k) { dot += pose[4*i+k]*pose[4*j+k]; }
            if (std::abs(dot - (i == j ? 1 : 0)) > 1e-3) {
                throw std::invalid_argument("Camera rotation is not orthonormal");
            }
        }
    }
    const float determinant = pose[0]*(pose[5]*pose[10]-pose[6]*pose[9]) -
        pose[1]*(pose[4]*pose[10]-pose[6]*pose[8]) + pose[2]*(pose[4]*pose[9]-pose[5]*pose[8]);
    if (std::abs(determinant-1) > 1e-3f) { throw std::invalid_argument("Camera rotation is reflected"); }
}

float PositiveOption(float value, const char* name) {
    if (!std::isfinite(value) || value <= 0) { throw std::invalid_argument(std::string(name)+" must be positive"); }
    return value;
}

// Independent double-precision residual reference used only by the optional
// diagnostic. For a single tangent coordinate, T*Exp(delta)*P is either an
// axis rotation of P or a translation of P before applying T. This is the
// right-multiplicative [rotation, translation] convention used by the solver.
std::array<double, 3> ReferencePixelResidual(
    const cunls::SE3Transform& pose, const cunls::Vector<3>& point,
    const std::vector<float>& focal, const std::vector<float>& pixel,
    const std::vector<float>& principal, int column, double step, double huber_delta) {
    std::array<double, 3> p = {point[0], point[1], point[2]};
    if (column >= 0 && column < 3) {
        std::array<double, 3> axis = {0, 0, 0};
        axis[column] = 1;
        const std::array<double, 3> cross = {
            axis[1]*p[2]-axis[2]*p[1], axis[2]*p[0]-axis[0]*p[2], axis[0]*p[1]-axis[1]*p[0]};
        const double along = p[column];
        for (int j = 0; j < 3; ++j) {
            p[j] = std::cos(step)*p[j] + std::sin(step)*cross[j] +
                   (1-std::cos(step))*axis[j]*along;
        }
    } else if (column >= 3 && column < 6) { p[column-3] += step; }
    else if (column >= 6 && column < 9) { p[column-6] += step; }
    std::array<double, 3> camera = {};
    for (int row = 0; row < 3; ++row) {
        camera[row] = static_cast<double>(pose[4*row+3]);
        for (int j = 0; j < 3; ++j) { camera[row] += static_cast<double>(pose[4*row+j])*p[j]; }
    }
    const double near = static_cast<double>(1e-4f);
    const double z = std::max(camera[2], near);
    const double fx = std::exp(static_cast<double>(focal[0]) + (column == 9 ? step : 0));
    const double fy = std::exp(static_cast<double>(focal[1]) + (column == 10 ? step : 0));
    const double u = fx*camera[0]/z + principal[0] + (column == 11 ? step : 0) - pixel[0];
    const double v = fy*camera[1]/z + principal[1] + (column == 12 ? step : 0) - pixel[1];
    const double radius = std::hypot(u, v);
    const double q = huber_delta > 0 && radius > huber_delta ? huber_delta/radius : 1;
    const double h = std::sqrt(q*(2-q));
    return {h*u, h*v, 1000*std::min(camera[2]-near, 0.0)};
}

// Compare the production CUDA Jacobian against double-precision differences.
float CheckPixelJacobian(const cunls::SE3Transform& pose, const cunls::Vector<3>& point,
                         const std::vector<float>& focal, const std::vector<float>& pixel,
                         const std::vector<float>& principal, cudaStream_t stream, float huber_delta) {
    cunls::dvector<cunls::SE3Transform> changed_pose(std::vector<cunls::SE3Transform>{pose});
    cunls::dvector<cunls::Vector<3>> changed_point(std::vector<cunls::Vector<3>>{point});
    cunls::dvector<float> changed_focal(focal), pixels(pixel), principal_device(principal);
    cunls::dvector<float> weight(std::vector<float>{1}), residual(3), jacobian(39);
    const std::vector<const float*> pointers = {reinterpret_cast<const float*>(changed_pose.data()),
        reinterpret_cast<const float*>(changed_point.data()), changed_focal.data(), principal_device.data()};
    cunls::dvector<const float*> links(pointers);
    PixelReprojectionFactors factor(pixels.data(), weight.data(), 1, huber_delta);
    if (!factor.Evaluate(residual.data(), jacobian.data(), links.data(), stream)) {
        throw std::runtime_error("Jacobian diagnostic evaluation failed");
    }
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    std::vector<float> analytic(39);
    jacobian.CopyToHost(analytic.data(), analytic.size());
    std::vector<float> cuda_residual(3);
    residual.CopyToHost(cuda_residual.data(), cuda_residual.size());
    const std::array<double, 3> reference = ReferencePixelResidual(
        pose, point, focal, pixel, principal, -1, 0, huber_delta);
    for (int row = 0; row < 3; ++row) {
        if (!std::isfinite(reference[row]) || !std::isfinite(cuda_residual[row]) ||
            std::abs(reference[row]-cuda_residual[row]) > 0.01 + 1e-4*std::abs(reference[row])) {
            throw std::runtime_error("CUDA residual disagrees with FP64 reference: " +
                std::string("row ")+std::to_string(row)+", CUDA "+std::to_string(cuda_residual[row])+", reference "+std::to_string(reference[row]));
        }
    }
    float maximum = 0;
    for (int column = 0; column < 13; ++column) {
        // FP64 avoids cancellation from perturbing/subtracting FP32 pixels.
        // Still require two successive passing, mutually consistent estimates.
        std::vector<double> previous(3, 0);
        bool previous_passed = false;
        bool accepted = false;
        double previous_error = 0;
        std::ostringstream attempts;
        for (int refinement = 0; refinement < 12; ++refinement) {
            const double epsilon = std::ldexp(1e-3, -refinement);
            const std::array<double, 3> plus = ReferencePixelResidual(
                pose, point, focal, pixel, principal, column, epsilon, huber_delta);
            const std::array<double, 3> minus = ReferencePixelResidual(
                pose, point, focal, pixel, principal, column, -epsilon, huber_delta);
            std::vector<double> numerical(3), normalized_errors(3);
            bool passed = true;
            bool stable = refinement > 0;
            double current_error = 0;
            for (int row = 0; row < 3; ++row) {
                numerical[row] = (plus[row]-minus[row])/(2*epsilon);
                const double exact = analytic[13*row+column];
                const double tolerance = 0.05 + 0.01*std::max(std::abs(numerical[row]), std::abs(exact));
                normalized_errors[row] = std::abs(numerical[row]-exact)/tolerance;
                passed = passed && std::isfinite(normalized_errors[row]) && normalized_errors[row] <= 1;
                stable = stable && std::isfinite(numerical[row]) &&
                    std::abs(numerical[row]-previous[row]) <= tolerance;
                current_error = std::max(current_error, normalized_errors[row]);
            }
            attempts << "epsilon=" << epsilon;
            for (int row = 0; row < 3; ++row) {
                attempts << " row=" << row << " numerical=" << numerical[row]
                         << " tolerance_ratio=" << normalized_errors[row];
            }
            attempts << ';';
            if (passed && previous_passed && stable) {
                maximum = std::max(maximum, static_cast<float>(std::max(current_error, previous_error)));
                accepted = true;
                break;
            }
            previous = numerical;
            previous_passed = passed;
            previous_error = current_error;
        }
        if (!accepted) {
            throw std::runtime_error("Reprojection Jacobian failed FP64 reference check: column " +
                std::to_string(column) + "; " + attempts.str());
        }
    }
    return maximum;
}

BAErrorStatistics ErrorStatistics(const std::vector<float>& errors, float c2) {
    std::vector<float> finite;
    std::size_t inliers = 0;
    double tls = 0;
    for (const float error : errors) {
        if (std::isfinite(error)) { finite.push_back(std::sqrt(error)); }
        if (error <= c2) { ++inliers; }
        tls += std::min(error, c2);
    }
    std::sort(finite.begin(), finite.end());
    return {errors.size(), finite.size(), inliers, tls,
            finite.empty() ? std::numeric_limits<float>::quiet_NaN() : finite[finite.size()/2]};
}
} // namespace

BundleAdjuster::BundleAdjuster(int device) : device_(device) {}

BAResult BundleAdjuster::Solve(const BAInput& input) const {
    int device_count = 0;
    THROW_ON_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    if (this->device_ < 0 || this->device_ >= device_count) { throw std::invalid_argument("CUDA device outside visible devices"); }
    THROW_ON_CUDA_ERROR(cudaSetDevice(this->device_));
    const BAOptions& options = input.options;
    const float threshold = PositiveOption(options.threshold_pixels, "threshold_pixels");
    const float focal_sigma = PositiveOption(options.focal_sigma_pixels, "focal_sigma_pixels");
    const float rotation_sigma = PositiveOption(options.rotation_sigma_radians, "rotation_sigma_radians");
    const float translation_sigma = PositiveOption(options.translation_sigma, "translation_sigma");
    const bool use_gnc = options.use_gnc;
    const float huber_delta = options.huber_delta_pixels;
    if (!std::isfinite(huber_delta) || huber_delta < 0 || (use_gnc && huber_delta > 0)) {
        throw std::invalid_argument("Huber delta must be nonnegative and used only without GNC");
    }
    const int rounds = use_gnc ? options.gnc_rounds : 1;
    const int iterations = options.lm_iterations;
    if (rounds < 1 || rounds > 128 || iterations < 1 || iterations > 1000) {
        throw std::invalid_argument("Invalid GNC/LM iteration limits");
    }
    const std::vector<BACamera>& cameras = input.cameras;
    const std::vector<std::array<float, 3>>& points = input.points;
    const std::vector<BAObservation>& observations = input.observations;
    if (cameras.size() < 3 || points.size() < 3 || observations.empty()) {
        throw std::invalid_argument("Need at least three cameras and three supported landmarks");
    }
    std::vector<cunls::SE3Transform> poses(cameras.size());
    std::vector<float> focals, log_focals, centers;
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        const std::vector<float> pose = std::vector<float>(cameras[i].world_to_camera.begin(), cameras[i].world_to_camera.end());
        ValidatePose(pose);
        std::copy(pose.begin(), pose.end(), poses[i].data());
        const std::vector<float> k = std::vector<float>(cameras[i].intrinsics.begin(), cameras[i].intrinsics.end());
        if (!std::all_of(k.begin(), k.end(), [](float v) { return std::isfinite(v); }) || k[0] <= 0 || k[1] <= 0) { throw std::invalid_argument("Intrinsics must be finite with positive focals"); }
        centers.insert(centers.end(), {k[2], k[3]});
        focals.insert(focals.end(), {k[0], k[1]});
        log_focals.insert(log_focals.end(), {std::log(k[0]), std::log(k[1])});
    }
    if (options.optimize_principal && !options.shared_intrinsics) {
        throw std::invalid_argument("Principal refinement requires shared calibration");
    }
    if (options.shared_intrinsics) {
        for (const BACamera& camera : cameras) {
            if (camera.intrinsics != cameras.front().intrinsics) {
                throw std::invalid_argument("Shared BA requires one initial calibration");
            }
        }
    }
    const std::size_t calibration_count = options.shared_intrinsics ? 1 : cameras.size();
    std::vector<float> center_states_values = centers;
    center_states_values.resize(2*calibration_count);
    log_focals.resize(2*calibration_count);
    std::vector<cunls::Vector<3>> positions(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        const std::vector<float> xyz = std::vector<float>(points[i].begin(), points[i].end());
        if (!std::all_of(xyz.begin(), xyz.end(), [](float v) { return std::isfinite(v); })) { throw std::invalid_argument("Nonfinite input point"); }
        std::copy(xyz.begin(), xyz.end(), positions[i].data());
    }
    std::vector<float> pixels, principal;
    std::vector<std::size_t> camera_ids, point_ids;
    std::vector<std::set<std::size_t>> views(points.size());
    std::vector<std::size_t> support(cameras.size(), 0);
    for (const BAObservation& observation : observations) {
        const std::size_t camera = observation.camera;
        const std::size_t point = observation.point;
        if (camera >= cameras.size() || point >= points.size() || !views[point].insert(camera).second) {
            throw std::invalid_argument("Unknown or duplicate camera/landmark observation");
        }
        ++support[camera];
        const std::vector<float> uv = std::vector<float>(observation.pixel.begin(), observation.pixel.end());
        if (!std::all_of(uv.begin(), uv.end(), [](float v) { return std::isfinite(v); })) { throw std::invalid_argument("Nonfinite input pixel"); }
        pixels.insert(pixels.end(), uv.begin(), uv.end());
        principal.push_back(cameras[camera].intrinsics[2]);
        principal.push_back(cameras[camera].intrinsics[3]);
        camera_ids.push_back(camera); point_ids.push_back(point);
    }
    std::set<std::size_t> connected = {0};
    bool changed = true;
    while (changed) {
        const std::size_t previous_size = connected.size();
        for (const std::set<std::size_t>& track : views) {
            if (track.size() < 3) { throw std::invalid_argument("Landmarks require three distinct views"); }
            bool intersects = false;
            for (const std::size_t camera : track) { intersects = intersects || connected.contains(camera); }
            if (intersects) { connected.insert(track.begin(), track.end()); }
        }
        changed = connected.size() != previous_size;
    }
    if (connected.size() != cameras.size() || *std::min_element(support.begin(), support.end()) < 6) {
        throw std::invalid_argument("Disconnected map or camera with fewer than six observations");
    }
    // Device arrays own their storage until after the problem and factors die.
    cunls::CudaStream stream;
    cunls::cuBLASHandle blas;
    cunls::dvector<cunls::SE3Transform> device_poses(poses), prior_poses(poses);
    cunls::dvector<cunls::Vector<3>> device_points(positions);
    cunls::dvector<float> device_focals(log_focals), prior_focals(focals);
    cunls::dvector<float> device_pixels(pixels);
    cunls::dvector<float> device_centers(center_states_values), prior_centers(centers);
    std::vector<int> fixed_centers(options.optimize_principal ? 0 : calibration_count);
    std::iota(fixed_centers.begin(),fixed_centers.end(),0);
    cunls::dvector<int> fixed_centers_device(fixed_centers);
    cunls::dvector<float> device_weights(std::vector<float>(observations.size(), 1.0f));
    cunls::dvector<float> device_errors(observations.size());
    cunls::dvector<int> constants(std::vector<int>{0});
    cunls::SE3StateBatch pose_states(blas, reinterpret_cast<const float*>(device_poses.data()),
                                   poses.size(), constants.data(), 1);
    cunls::VectorStateBatch<3> point_states(reinterpret_cast<const float*>(device_points.data()), positions.size());
    cunls::VectorStateBatch<2> focal_states(device_focals.data(), calibration_count);
    cunls::VectorStateBatch<2> center_states(device_centers.data(), calibration_count, fixed_centers_device.data(), fixed_centers.size());
    PixelReprojectionFactors reprojection(device_pixels.data(), device_weights.data(), observations.size(), huber_delta);
    PosePriorFactors pose_priors(prior_poses.data(), poses.size(), rotation_sigma, translation_sigma);
    CalibrationPriorFactors focal_priors(prior_focals.data(), poses.size(), focal_sigma);
    CalibrationPriorFactors center_priors(prior_centers.data(), poses.size(),
        PositiveOption(options.principal_sigma_pixels,"principal_sigma_pixels"), false);
    std::vector<float*> links, pose_links, focal_links, center_links;
    for (std::size_t i = 0; i < observations.size(); ++i) {
        links.push_back(pose_states.StateBlockDevicePtr(camera_ids[i]));
        links.push_back(point_states.StateBlockDevicePtr(point_ids[i]));
        links.push_back(focal_states.StateBlockDevicePtr(options.shared_intrinsics ? 0 : camera_ids[i]));
        links.push_back(center_states.StateBlockDevicePtr(options.shared_intrinsics ? 0 : camera_ids[i]));
    }
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        pose_links.push_back(pose_states.StateBlockDevicePtr(i));
        focal_links.push_back(focal_states.StateBlockDevicePtr(options.shared_intrinsics ? 0 : i));
        center_links.push_back(center_states.StateBlockDevicePtr(options.shared_intrinsics ? 0 : i));
    }
    const std::vector<const float*> const_links(links.begin(), links.end());
    cunls::dvector<const float*> device_links(const_links);
    cunls::LevenbergMarquardtMinimizerOptions lm;
    lm.base_options.max_num_iterations = static_cast<std::size_t>(iterations);
    lm.base_options.sparse_linear_solver_type = cunls::SparseLinearSolverType::cuDSS;
    std::vector<float> errors(observations.size()), weights(observations.size(), 1);
    EvaluatePixelErrors(device_pixels.data(), device_links.data(),
                        device_errors.data(), observations.size(), stream.GetStream());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    device_errors.CopyToHost(errors.data(), errors.size());
    if (!std::all_of(errors.begin(), errors.end(), [](float x) { return std::isfinite(x); })) {
        throw std::invalid_argument("Initial observations must have positive finite depth and projection");
    }
    const float c2 = threshold*threshold;
    const float maximum = *std::max_element(errors.begin(), errors.end());
    float mu = std::max(1e-6f, c2 / std::max(c2, 2*maximum-c2));
    BAReport report;
    report.before = ErrorStatistics(errors, c2);
    report.use_gnc = use_gnc;
    report.huber_delta_pixels = huber_delta;
    if (options.check_jacobians) {
        float maximum_error = 0;
        const std::size_t samples = std::min<std::size_t>(8, observations.size());
        for (std::size_t sample = 0; sample < samples; ++sample) {
            const std::size_t i = sample*observations.size()/samples;
            const std::size_t camera = camera_ids[i];
            const std::size_t calibration = options.shared_intrinsics ? 0 : camera;
            maximum_error = std::max(maximum_error, CheckPixelJacobian(poses[camera], positions[point_ids[i]],
                {log_focals[2*calibration], log_focals[2*calibration+1]}, {pixels[2*i], pixels[2*i+1]},
                {principal[2*i], principal[2*i+1]}, stream.GetStream(), huber_delta));
            if (huber_delta > 0) {
                // Exercise the robust branch even when all sampled data fit
                // inside the quadratic region. This changes diagnostics only.
                maximum_error = std::max(maximum_error, CheckPixelJacobian(poses[camera], positions[point_ids[i]],
                    {log_focals[2*calibration], log_focals[2*calibration+1]},
                    {pixels[2*i] + 10*huber_delta, pixels[2*i+1]},
                    {principal[2*i], principal[2*i+1]}, stream.GetStream(), huber_delta));
            }
        }
        report.jacobian_samples = samples;
        report.maximum_jacobian_tolerance_ratio = maximum_error;
    }
    double previous_tls = report.before.tls_cost;
    for (int round = 0; round < rounds; ++round) {
        if (use_gnc) {
            UpdateTlsWeights(device_errors.data(), device_weights.data(), observations.size(), c2, mu, stream.GetStream());
        }
        THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
        device_weights.CopyToHost(weights.data(), weights.size());
        std::vector<std::size_t> active_views(positions.size(), 0);
        for (std::size_t i = 0; i < weights.size(); ++i) {
            if (weights[i] > 0.01f) { ++active_views[point_ids[i]]; }
        }
        std::vector<int> frozen;
        for (std::size_t i = 0; i < active_views.size(); ++i) {
            if (active_views[i] < 3) { frozen.push_back(static_cast<int>(i)); }
        }
        // GNC can remove all constraints on a landmark. Freeze such states for
        // this round instead of adding fake observations or a nonzero TLS floor.
        // Rebuild the problem so the solver sees the updated constant mask.
        cunls::dvector<int> frozen_device(frozen);
        cunls::VectorStateBatch<3> round_points(reinterpret_cast<const float*>(device_points.data()),
            positions.size(), frozen_device.data(), frozen.size());
        cunls::Problem problem;
        problem.AddStateBatch(&pose_states); problem.AddStateBatch(&round_points); problem.AddStateBatch(&focal_states); problem.AddStateBatch(&center_states);
        problem.AddFactorBatch(&reprojection, links);
        problem.AddFactorBatch(&pose_priors, pose_links); problem.AddFactorBatch(&focal_priors, focal_links);
        if (options.optimize_principal) { problem.AddFactorBatch(&center_priors, center_links); }
        if (!problem.CheckConsistency()) { throw std::runtime_error("cuNLS graph consistency failed"); }
        cunls::LevenbergMarquardtMinimizer minimizer(lm);
        const cunls::MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
        EvaluatePixelErrors(device_pixels.data(), device_links.data(),
                            device_errors.data(), observations.size(), stream.GetStream());
        THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
        if (!std::isfinite(summary.final_cost) || !std::isfinite(summary.initial_cost) ||
            summary.final_cost > summary.initial_cost + 1e-4f*std::max(1.0f, summary.initial_cost)) {
            throw std::runtime_error("cuNLS returned nonfinite or increasing weighted cost");
        }
        device_errors.CopyToHost(errors.data(), errors.size());
        device_weights.CopyToHost(weights.data(), weights.size());
        const BAErrorStatistics stats = ErrorStatistics(errors, c2);
        const double tls = stats.tls_cost;
        const std::size_t soft = static_cast<std::size_t>(std::count_if(weights.begin(), weights.end(),
            [](float w) { return w > 0.01f && w < 0.99f; }));
        report.rounds.push_back({mu, summary.num_iterations, summary.initial_cost, summary.final_cost,
                                 soft, frozen.size(), stats});
        if (!use_gnc) {
            // The joint solve uses fixed external weights; optional Huber
            // robustification is evaluated inside the pixel factor.
            // A budget-limited LM solve is retained, but not called complete.
            report.optimization_complete = summary.num_iterations < static_cast<std::size_t>(iterations);
            report.lm_budget_exhausted = summary.num_iterations >= static_cast<std::size_t>(iterations);
            break;
        }
        if (options.verbose) { std::cerr << "GNC " << round+1 << "/" << rounds << ": " << stats.within_threshold
                  << "/" << observations.size() << " observations within " << threshold << " px\n"; }
        if (mu >= 1 && soft == 0 && std::abs(tls-previous_tls) <= 1e-5*std::max(1.0, previous_tls)) {
            report.gnc_converged = true; report.optimization_complete = true; break;
        }
        previous_tls = tls;
        mu = std::min(mu*1.6f, 1e6f);
    }
    device_poses.CopyToHost(poses.data(), poses.size());
    device_points.CopyToHost(positions.data(), positions.size());
    device_focals.CopyToHost(log_focals.data(), log_focals.size());
    device_centers.CopyToHost(center_states_values.data(),center_states_values.size());
    BAResult output;
    for (std::size_t i = 0; i < poses.size(); ++i) {
        const std::size_t calibration = options.shared_intrinsics ? 0 : i;
        const std::vector<float> pose(poses[i].begin(), poses[i].end());
        ValidatePose(pose);
        if (!std::all_of(pose.begin(), pose.end(), [](float v) { return std::isfinite(v); }) ||
            !std::isfinite(std::exp(log_focals[2*calibration])) || !std::isfinite(std::exp(log_focals[2*calibration+1])) ||
            std::exp(log_focals[2*calibration]) <= 0 || std::exp(log_focals[2*calibration+1]) <= 0) {
            throw std::runtime_error("Invalid optimized camera");
        }
        BACamera camera;
        std::copy(pose.begin(), pose.end(), camera.world_to_camera.begin());
        camera.intrinsics = {std::exp(log_focals[2*calibration]), std::exp(log_focals[2*calibration+1]),
                             center_states_values[2*calibration], center_states_values[2*calibration+1]};
        if (!std::isfinite(camera.intrinsics[2]) || !std::isfinite(camera.intrinsics[3])) { throw std::runtime_error("Nonfinite principal point"); }
        output.cameras.push_back(camera);
    }
    for (const cunls::Vector<3>& point : positions) {
        if (!std::all_of(point.begin(), point.end(), [](float v) { return std::isfinite(v); })) {
            throw std::runtime_error("Invalid optimized point");
        }
        output.points.push_back({point[0], point[1], point[2]});
    }
    report.after = ErrorStatistics(errors, c2);
    output.report = std::move(report);
    return output;
}
} // namespace sceneforge::optimization
