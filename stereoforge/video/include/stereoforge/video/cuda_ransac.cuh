#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace stereoforge::video {
inline constexpr int ransac_hypotheses = 2000;
inline constexpr int ransac_models = 2 * ransac_hypotheses + 2;
inline constexpr int maximum_keypoints = 4096;
struct DeviceMatch final {
    float ax, ay, bx, by;
    int source, target;
    float confidence;
};
struct RansacModel final {
    double matrix[9];
    int inliers;
    float cost;
    int homography;
    int valid;
};
struct RansacSummary final {
    int matches, inliers, homography, invalid;
    float coverage, displacement;
};
// All pointers except dimensions/settings refer to caller-owned device memory.
// Enqueued on the matcher stream; no host readback or allocation between stages.
void verifyDeviceMatches(const void* points0, const void* scores0, const void* points1,
    const void* scores1, const std::int32_t* indices, const void* confidence, bool half,
    int count, int width0, int height0, int width1, int height1,
    float detector_threshold, float match_threshold, float pixel_threshold,
    unsigned seed, DeviceMatch* matches, int* match_count, RansacModel* models,
    int* winners, unsigned char* mask, RansacSummary* summary, cudaStream_t stream);
}  // namespace stereoforge::video
