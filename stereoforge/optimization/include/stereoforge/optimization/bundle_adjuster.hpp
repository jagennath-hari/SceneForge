#pragma once
#include <array>
#include <vector>
#include <cstddef>
#include <limits>

namespace stereoforge::optimization {
struct BACamera {
    std::array<float, 16> world_to_camera;
    std::array<float, 4> intrinsics;
};
struct BAObservation {
    std::size_t camera;
    std::size_t point;
    std::array<float, 2> pixel;
};
struct BAOptions {
    float threshold_pixels = 3;
    float focal_sigma_pixels = 10;
    float rotation_sigma_radians = 0.1f;
    float translation_sigma = 0.1f;
    float huber_delta_pixels = 0;
    int gnc_rounds = 64;
    int lm_iterations = 50;
    bool use_gnc = true;
    bool check_jacobians = false;
    bool verbose = false;
};
struct BAErrorStatistics {
    std::size_t observations = 0;
    std::size_t positive_finite = 0;
    std::size_t within_threshold = 0;
    double tls_cost = 0;
    float median_pixels = std::numeric_limits<float>::quiet_NaN();
};
struct BARound {
    float mu;
    std::size_t lm_iterations;
    double weighted_cost_before;
    double weighted_cost_after;
    std::size_t soft_weights;
    std::size_t frozen_landmarks;
    BAErrorStatistics errors;
};
struct BAReport {
    BAErrorStatistics before;
    BAErrorStatistics after;
    std::vector<BARound> rounds;
    bool gnc_converged = false;
    bool optimization_complete = false;
    bool lm_budget_exhausted = false;
    bool use_gnc = true;
    float huber_delta_pixels = 0;
    std::size_t jacobian_samples = 0;
    float maximum_jacobian_tolerance_ratio = 0;
};
struct BAInput {
    std::vector<BACamera> cameras;
    std::vector<std::array<float, 3>> points;
    std::vector<BAObservation> observations;
    BAOptions options;
};
struct BAResult {
    std::vector<BACamera> cameras;
    std::vector<std::array<float, 3>> points;
    BAReport report;
};
class BundleAdjuster final {
public:
    explicit BundleAdjuster(int device);
    [[nodiscard]] BAResult Solve(const BAInput& input) const;
private:
    int device_;
};
} // namespace stereoforge::optimization
