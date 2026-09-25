#pragma once
#include <array>
#include <vector>
#include <cstddef>
#include <nlohmann/json.hpp>

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
struct BAInput {
    std::vector<BACamera> cameras;
    std::vector<std::array<float, 3>> points;
    std::vector<BAObservation> observations;
    nlohmann::json options;
};
struct BAResult {
    std::vector<BACamera> cameras;
    std::vector<std::array<float, 3>> points;
    nlohmann::json report;
};
class BundleAdjuster final {
public:
    explicit BundleAdjuster(int device);
    [[nodiscard]] BAResult Solve(const BAInput& input) const;
    // Compatibility for the standalone solver; the map builder uses typed buffers.
    [[nodiscard]] nlohmann::json Solve(const nlohmann::json& input) const;
private:
    int device_;
};
} // namespace stereoforge::optimization
