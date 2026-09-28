#include "stereoforge/optimization/bundle_adjuster.hpp"
#include <filesystem>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace stereoforge::optimization;
using Json = nlohmann::json;

std::vector<float> ReadVector(const Json& value, std::size_t size) {
    const std::vector<float> result = value.get<std::vector<float>>();
    if (result.size() != size || !std::all_of(result.begin(), result.end(),
        [](float v) { return std::isfinite(v); })) {
        throw std::invalid_argument("Invalid vector length or nonfinite input");
    }
    return result;
}


Json StatisticsJson(const BAErrorStatistics& stats) {
    return {{"observations", stats.observations}, {"positive_finite", stats.positive_finite},
            {"within_threshold", stats.within_threshold}, {"tls_cost", stats.tls_cost},
            {"median_pixels", std::isfinite(stats.median_pixels) ? Json(stats.median_pixels) : Json(nullptr)}};
}
Json ReportJson(const BAReport& report) {
    Json result = {{"before", StatisticsJson(report.before)}, {"after", StatisticsJson(report.after)},
        {"gnc_converged", report.gnc_converged}, {"optimization_complete", report.optimization_complete},
        {"lm_budget_exhausted", report.lm_budget_exhausted}, {"use_gnc", report.use_gnc},
        {"huber_delta_pixels", report.huber_delta_pixels}, {"solver", "cuNLS LM/cuDSS"}, {"rounds", Json::array()}};
    for (const BARound& round : report.rounds) {
        result["rounds"].push_back({{"mu", round.mu}, {"lm_iterations", round.lm_iterations},
            {"weighted_cost_before", round.weighted_cost_before}, {"weighted_cost_after", round.weighted_cost_after},
            {"iteration_costs", round.iteration_costs}, {"soft_weights", round.soft_weights}, {"frozen_landmarks", round.frozen_landmarks}, {"errors", StatisticsJson(round.errors)}});
    }
    if (report.jacobian_samples) {
        result["jacobian_check"] = {{"observations", report.jacobian_samples},
            {"maximum_tolerance_ratio", report.maximum_jacobian_tolerance_ratio}, {"passed", true}};
    }
    return result;
}
Json SolveFileRequest(const Json& input, int device) {
    if (input.at("format_version") != 1) { throw std::invalid_argument("Unsupported input format"); }
    BAInput request;
    const Json options = input.value("options", Json::object());
    request.options.threshold_pixels = options.value("threshold_pixels", request.options.threshold_pixels);
    request.options.focal_sigma_pixels = options.value("focal_sigma_pixels", request.options.focal_sigma_pixels);
    request.options.rotation_sigma_radians = options.value("rotation_sigma_radians", request.options.rotation_sigma_radians);
    request.options.translation_sigma = options.value("translation_sigma", request.options.translation_sigma);
    request.options.huber_delta_pixels = options.value("huber_delta_pixels", request.options.huber_delta_pixels);
    request.options.gnc_rounds = options.value("gnc_rounds", request.options.gnc_rounds);
    request.options.lm_iterations = options.value("lm_iterations", request.options.lm_iterations);
    request.options.use_gnc = options.value("use_gnc", request.options.use_gnc);
    request.options.check_jacobians = options.value("check_jacobians", request.options.check_jacobians);
    request.options.verbose = options.value("verbose", request.options.verbose);

    for (const Json& value : input.at("cameras")) {
        BACamera camera;
        const std::vector<float> pose = ReadVector(value.at("world_to_camera"), 16);
        const std::vector<float> k = ReadVector(value.at("intrinsics"), 4);
        std::copy(pose.begin(), pose.end(), camera.world_to_camera.begin());
        std::copy(k.begin(), k.end(), camera.intrinsics.begin());
        request.cameras.push_back(camera);
    }
    for (const Json& value : input.at("points")) {
        const std::vector<float> xyz = ReadVector(value, 3);
        request.points.push_back({xyz[0], xyz[1], xyz[2]});
    }
    for (const Json& value : input.at("observations")) {
        const std::vector<float> uv = ReadVector(value.at("pixel"), 2);
        request.observations.push_back({value.at("camera").get<std::size_t>(),
            value.at("point").get<std::size_t>(), {uv[0], uv[1]}});
    }
    const BAResult result = BundleAdjuster(device).Solve(request);
    Json output = {{"format_version", 1}, {"cameras", Json::array()}, {"points", result.points}, {"report", ReportJson(result.report)}};
    for (const BACamera& camera : result.cameras) {
        output["cameras"].push_back({{"world_to_camera", camera.world_to_camera}, {"intrinsics", camera.intrinsics}});
    }
    return output;
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: stereoforge-bundle-adjust INPUT.json OUTPUT.json DEVICE\n";
            return 2;
        }
        const std::filesystem::path output(argv[2]);
        const std::filesystem::path temporary(output.string()+".partial");
        if (std::filesystem::exists(output) || std::filesystem::exists(temporary)) {
            throw std::runtime_error("Output already exists; use a fresh diagnostic directory");
        }
        std::ifstream input(argv[1]);
        input.exceptions(std::ios::badbit);
        if (!input) { throw std::runtime_error("Cannot open input JSON"); }
        nlohmann::json request;
        input >> request;
        const nlohmann::json result = SolveFileRequest(request, std::stoi(argv[3]));
        std::ofstream stream(temporary);
        stream.exceptions(std::ios::failbit | std::ios::badbit);
        stream << result.dump() << '\n';
        stream.close();
        std::filesystem::rename(temporary, output);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cuNLS BA failed: " << error.what() << '\n';
        return 1;
    }
}
