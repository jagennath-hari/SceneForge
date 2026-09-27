#pragma once
#include "stereoforge/optimization/map_builder.hpp"

namespace stereoforge::optimization {
[[nodiscard]] std::vector<SparseMap> ConnectedGroups(const SparseMap& model);

// Recover disconnected initialization components from original measured tracks.
// Operates only on the provisional window; the accepted global map is read-only.
void RecoverInitializationBridges(SparseMap& model, const std::vector<Observations>& tracks,
    const std::vector<DepthFrame>& frames, const SparseMap& accepted,
    const std::function<void(const std::string&)>& progress);
} // namespace stereoforge::optimization
