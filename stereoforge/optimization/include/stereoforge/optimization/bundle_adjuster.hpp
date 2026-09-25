#pragma once
#include <nlohmann/json.hpp>

namespace stereoforge::optimization {
class BundleAdjuster final {
public:
    explicit BundleAdjuster(int device);
    // JSON boundary is for a bounded diagnostic; all optimization runs natively.
    [[nodiscard]] nlohmann::json Solve(const nlohmann::json& input) const;
private:
    int device_;
};
} // namespace stereoforge::optimization
