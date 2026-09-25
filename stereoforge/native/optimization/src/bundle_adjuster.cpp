#include "stereoforge/optimization/bundle_adjuster.hpp"
#include "stereoforge/optimization/factors.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
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

namespace stereoforge::optimization {
// cuNLS error macros expand an unqualified LogError at the call site.
using cunls::LogError;

namespace {
using Json = nlohmann::json;

std::vector<float> ReadVector(const Json& value, std::size_t size) {
    const std::vector<float> result = value.get<std::vector<float>>();
    if (result.size() != size || !std::all_of(result.begin(), result.end(),
        [](float v) { return std::isfinite(v); })) {
        throw std::invalid_argument("Invalid vector length or nonfinite input");
    }
    return result;
}

void ValidatePose(const std::vector<float>& pose) {
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

float PositiveOption(const Json& options, const char* name, float fallback) {
    const float value = options.value(name, fallback);
    if (!std::isfinite(value) || value <= 0) { throw std::invalid_argument(std::string(name)+" must be positive"); }
    return value;
}

// Optional runtime diagnostic: compare the analytic reprojection Jacobian with
// central differences through cuNLS's actual SE3 Plus operation. No solver run.
float CheckPixelJacobian(const cunls::SE3Transform& pose, const cunls::Vector<3>& point,
                         const std::vector<float>& focal, const std::vector<float>& pixel,
                         const std::vector<float>& principal, cudaStream_t stream,
                         cunls::cuBLASHandle& blas) {
    cunls::dvector<cunls::SE3Transform> base_pose(std::vector<cunls::SE3Transform>{pose});
    cunls::dvector<cunls::SE3Transform> changed_pose(std::vector<cunls::SE3Transform>{pose});
    cunls::dvector<cunls::Vector<3>> changed_point(std::vector<cunls::Vector<3>>{point});
    cunls::dvector<float> changed_focal(focal), pixels(pixel), principal_device(principal);
    cunls::dvector<float> weight(std::vector<float>{1}), residual(3), jacobian(33), delta(6);
    cunls::SE3StateBatch state(blas, reinterpret_cast<const float*>(base_pose.data()), 1);
    const std::vector<const float*> pointers = {reinterpret_cast<const float*>(changed_pose.data()),
        reinterpret_cast<const float*>(changed_point.data()), changed_focal.data()};
    cunls::dvector<const float*> links(pointers);
    PixelReprojectionFactors factor(pixels.data(), principal_device.data(), weight.data(), 1);
    if (!factor.Evaluate(residual.data(), jacobian.data(), links.data(), stream)) {
        throw std::runtime_error("Jacobian diagnostic evaluation failed");
    }
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    std::vector<float> analytic(33);
    jacobian.CopyToHost(analytic.data(), analytic.size());
    float maximum = 0;
    for (int column = 0; column < 11; ++column) {
        const float epsilon = 1e-3f;
        std::vector<float> plus(3), minus(3);
        for (const int sign : {-1, 1}) {
            changed_pose.CopyFromHost(&pose, 1);
            cunls::Vector<3> perturbed = point;
            std::vector<float> perturbed_focal = focal;
            if (column < 6) {
                std::vector<float> twist(6, 0);
                twist[column] = sign*epsilon;
                delta.CopyFromHost(twist.data(), 6);
                state.Plus(reinterpret_cast<const float*>(base_pose.data()), delta.data(),
                            reinterpret_cast<float*>(changed_pose.data()), stream);
            } else if (column < 9) { perturbed[column-6] += sign*epsilon; }
            else { perturbed_focal[column-9] += sign*epsilon; }
            changed_point.CopyFromHost(&perturbed, 1);
            changed_focal.CopyFromHost(perturbed_focal.data(), 2);
            if (!factor.Evaluate(residual.data(), nullptr, links.data(), stream)) {
                throw std::runtime_error("Finite-difference diagnostic evaluation failed");
            }
            THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
            residual.CopyToHost(sign > 0 ? plus.data() : minus.data(), 3);
        }
        for (int row = 0; row < 3; ++row) {
            const float numerical = (plus[row]-minus[row])/(2*epsilon);
            const float exact = analytic[11*row+column];
            const float error = std::abs(numerical-exact)/(0.05f + 0.01f*std::max(std::abs(numerical), std::abs(exact)));
            if (!std::isfinite(error) || error > 1) {
                throw std::runtime_error("Reprojection Jacobian failed central-difference check");
            }
            maximum = std::max(maximum, error);
        }
    }
    return maximum;
}

Json ErrorStatistics(const std::vector<float>& errors, float c2) {
    std::vector<float> finite;
    std::size_t inliers = 0;
    double tls = 0;
    for (const float error : errors) {
        if (std::isfinite(error)) { finite.push_back(std::sqrt(error)); }
        if (error <= c2) { ++inliers; }
        tls += std::min(error, c2);
    }
    std::sort(finite.begin(), finite.end());
    return {{"observations", errors.size()}, {"positive_finite", finite.size()},
            {"within_threshold", inliers}, {"tls_cost", tls},
            {"median_pixels", finite.empty() ? Json(nullptr) : Json(finite[finite.size()/2])}};
}
} // namespace

BundleAdjuster::BundleAdjuster(int device) : device_(device) {}

nlohmann::json BundleAdjuster::Solve(const nlohmann::json& input) const {
    if (input.at("format_version") != 1) { throw std::invalid_argument("Unsupported input format"); }
    int device_count = 0;
    THROW_ON_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    if (this->device_ < 0 || this->device_ >= device_count) { throw std::invalid_argument("CUDA device outside visible devices"); }
    THROW_ON_CUDA_ERROR(cudaSetDevice(this->device_));
    const Json options = input.value("options", Json::object());
    const float threshold = PositiveOption(options, "threshold_pixels", 3);
    const float focal_sigma = PositiveOption(options, "focal_sigma_pixels", 10);
    const float rotation_sigma = PositiveOption(options, "rotation_sigma_radians", 0.1f);
    const float translation_sigma = PositiveOption(options, "translation_sigma", 0.1f);
    const int rounds = options.value("gnc_rounds", 64);
    const int iterations = options.value("lm_iterations", 50);
    if (rounds < 1 || rounds > 128 || iterations < 1 || iterations > 500) {
        throw std::invalid_argument("Invalid GNC/LM iteration limits");
    }
    const Json& cameras = input.at("cameras");
    const Json& points = input.at("points");
    const Json& observations = input.at("observations");
    if (cameras.size() < 3 || points.size() < 3 || observations.empty()) {
        throw std::invalid_argument("Need at least three cameras and three supported landmarks");
    }
    std::vector<cunls::SE3Transform> poses(cameras.size());
    std::vector<float> focals, log_focals;
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        const std::vector<float> pose = ReadVector(cameras[i].at("world_to_camera"), 16);
        ValidatePose(pose);
        std::copy(pose.begin(), pose.end(), poses[i].data());
        const std::vector<float> k = ReadVector(cameras[i].at("intrinsics"), 4);
        if (k[0] <= 0 || k[1] <= 0) { throw std::invalid_argument("Focals must be positive"); }
        focals.insert(focals.end(), {k[0], k[1]});
        log_focals.insert(log_focals.end(), {std::log(k[0]), std::log(k[1])});
    }
    std::vector<cunls::Vector<3>> positions(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        const std::vector<float> xyz = ReadVector(points[i], 3);
        std::copy(xyz.begin(), xyz.end(), positions[i].data());
    }
    std::vector<float> pixels, principal;
    std::vector<std::size_t> camera_ids, point_ids;
    std::vector<std::set<std::size_t>> views(points.size());
    std::vector<std::size_t> support(cameras.size(), 0);
    for (const Json& observation : observations) {
        const std::size_t camera = observation.at("camera").get<std::size_t>();
        const std::size_t point = observation.at("point").get<std::size_t>();
        if (camera >= cameras.size() || point >= points.size() || !views[point].insert(camera).second) {
            throw std::invalid_argument("Unknown or duplicate camera/landmark observation");
        }
        ++support[camera];
        const std::vector<float> uv = ReadVector(observation.at("pixel"), 2);
        pixels.insert(pixels.end(), uv.begin(), uv.end());
        principal.push_back(cameras[camera].at("intrinsics").at(2).get<float>());
        principal.push_back(cameras[camera].at("intrinsics").at(3).get<float>());
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
    cunls::dvector<float> device_pixels(pixels), device_principal(principal);
    cunls::dvector<float> device_weights(std::vector<float>(observations.size(), 1.0f));
    cunls::dvector<float> device_errors(observations.size());
    cunls::dvector<int> constants(std::vector<int>{0});
    cunls::SE3StateBatch pose_states(blas, reinterpret_cast<const float*>(device_poses.data()),
                                   poses.size(), constants.data(), 1);
    cunls::VectorStateBatch<3> point_states(reinterpret_cast<const float*>(device_points.data()), positions.size());
    cunls::VectorStateBatch<2> focal_states(device_focals.data(), cameras.size());
    PixelReprojectionFactors reprojection(device_pixels.data(), device_principal.data(), device_weights.data(), observations.size());
    PosePriorFactors pose_priors(prior_poses.data(), poses.size(), rotation_sigma, translation_sigma);
    FocalPriorFactors focal_priors(prior_focals.data(), poses.size(), focal_sigma);
    std::vector<float*> links, pose_links, focal_links;
    for (std::size_t i = 0; i < observations.size(); ++i) {
        links.push_back(pose_states.StateBlockDevicePtr(camera_ids[i]));
        links.push_back(point_states.StateBlockDevicePtr(point_ids[i]));
        links.push_back(focal_states.StateBlockDevicePtr(camera_ids[i]));
    }
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        pose_links.push_back(pose_states.StateBlockDevicePtr(i));
        focal_links.push_back(focal_states.StateBlockDevicePtr(i));
    }
    const std::vector<const float*> const_links(links.begin(), links.end());
    cunls::dvector<const float*> device_links(const_links);
    cunls::LevenbergMarquardtMinimizerOptions lm;
    lm.base_options.max_num_iterations = static_cast<std::size_t>(iterations);
    lm.base_options.sparse_linear_solver_type = cunls::SparseLinearSolverType::cuDSS;
    std::vector<float> errors(observations.size()), weights(observations.size(), 1);
    EvaluatePixelErrors(device_pixels.data(), device_principal.data(), device_links.data(),
                        device_errors.data(), observations.size(), stream.GetStream());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    device_errors.CopyToHost(errors.data(), errors.size());
    if (!std::all_of(errors.begin(), errors.end(), [](float x) { return std::isfinite(x); })) {
        throw std::invalid_argument("Initial observations must have positive finite depth and projection");
    }
    const float c2 = threshold*threshold;
    const float maximum = *std::max_element(errors.begin(), errors.end());
    float mu = std::max(1e-6f, c2 / std::max(c2, 2*maximum-c2));
    Json report = {{"before", ErrorStatistics(errors, c2)}, {"rounds", Json::array()},
                   {"gnc_converged", false}, {"solver", "cuNLS LM/cuDSS"}};
    if (options.value("check_jacobians", false)) {
        float maximum_error = 0;
        const std::size_t samples = std::min<std::size_t>(8, observations.size());
        for (std::size_t sample = 0; sample < samples; ++sample) {
            const std::size_t i = sample*observations.size()/samples;
            const std::size_t camera = camera_ids[i];
            maximum_error = std::max(maximum_error, CheckPixelJacobian(poses[camera], positions[point_ids[i]],
                {log_focals[2*camera], log_focals[2*camera+1]}, {pixels[2*i], pixels[2*i+1]},
                {principal[2*i], principal[2*i+1]}, stream.GetStream(), blas));
        }
        report["jacobian_check"] = {{"observations", samples}, {"maximum_tolerance_ratio", maximum_error},
            {"passed", true}, {"scope", "pixel factor, all 11 tangent columns; initial positive-depth states"}};
    }
    double previous_tls = report["before"]["tls_cost"].get<double>();
    for (int round = 0; round < rounds; ++round) {
        UpdateTlsWeights(device_errors.data(), device_weights.data(), observations.size(), c2, mu, stream.GetStream());
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
        problem.AddStateBatch(&pose_states); problem.AddStateBatch(&round_points); problem.AddStateBatch(&focal_states);
        problem.AddFactorBatch(&reprojection, links);
        problem.AddFactorBatch(&pose_priors, pose_links); problem.AddFactorBatch(&focal_priors, focal_links);
        if (!problem.CheckConsistency()) { throw std::runtime_error("cuNLS graph consistency failed"); }
        cunls::LevenbergMarquardtMinimizer minimizer(lm);
        const cunls::MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
        EvaluatePixelErrors(device_pixels.data(), device_principal.data(), device_links.data(),
                            device_errors.data(), observations.size(), stream.GetStream());
        THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
        if (!std::isfinite(summary.final_cost) || !std::isfinite(summary.initial_cost) ||
            summary.final_cost > summary.initial_cost + 1e-4f*std::max(1.0f, summary.initial_cost)) {
            throw std::runtime_error("cuNLS returned nonfinite or increasing weighted cost");
        }
        device_errors.CopyToHost(errors.data(), errors.size());
        device_weights.CopyToHost(weights.data(), weights.size());
        const Json stats = ErrorStatistics(errors, c2);
        const double tls = stats["tls_cost"].get<double>();
        const std::size_t soft = static_cast<std::size_t>(std::count_if(weights.begin(), weights.end(),
            [](float w) { return w > 0.01f && w < 0.99f; }));
        report["rounds"].push_back({{"mu", mu}, {"lm_iterations", summary.num_iterations},
            {"weighted_cost_before", summary.initial_cost}, {"weighted_cost_after", summary.final_cost},
            {"soft_weights", soft}, {"frozen_landmarks", frozen.size()}, {"errors", stats}});
        std::cerr << "GNC " << round+1 << "/" << rounds << ": " << stats["within_threshold"]
                  << "/" << observations.size() << " observations within " << threshold << " px\n";
        if (mu >= 1 && soft == 0 && std::abs(tls-previous_tls) <= 1e-5*std::max(1.0, previous_tls)) {
            report["gnc_converged"] = true; break;
        }
        previous_tls = tls;
        mu = std::min(mu*1.6f, 1e6f);
    }
    device_poses.CopyToHost(poses.data(), poses.size());
    device_points.CopyToHost(positions.data(), positions.size());
    device_focals.CopyToHost(log_focals.data(), log_focals.size());
    Json output = {{"format_version", 1}, {"cameras", Json::array()}, {"points", Json::array()}};
    for (std::size_t i = 0; i < poses.size(); ++i) {
        const std::vector<float> pose(poses[i].begin(), poses[i].end());
        ValidatePose(pose);
        if (!std::all_of(pose.begin(), pose.end(), [](float v) { return std::isfinite(v); }) ||
            !std::isfinite(std::exp(log_focals[2*i])) || !std::isfinite(std::exp(log_focals[2*i+1])) ||
            std::exp(log_focals[2*i]) <= 0 || std::exp(log_focals[2*i+1]) <= 0) {
            throw std::runtime_error("Invalid optimized camera");
        }
        output["cameras"].push_back({{"world_to_camera", pose},
            {"intrinsics", {std::exp(log_focals[2*i]), std::exp(log_focals[2*i+1]),
                             cameras[i]["intrinsics"][2].get<float>(), cameras[i]["intrinsics"][3].get<float>()}}});
    }
    for (const cunls::Vector<3>& point : positions) {
        if (!std::all_of(point.begin(), point.end(), [](float v) { return std::isfinite(v); })) {
            throw std::runtime_error("Invalid optimized point");
        }
        output["points"].push_back(std::vector<float>(point.begin(), point.end()));
    }
    report["after"] = ErrorStatistics(errors, c2);
    output["report"] = report;
    return output;
}
} // namespace stereoforge::optimization
