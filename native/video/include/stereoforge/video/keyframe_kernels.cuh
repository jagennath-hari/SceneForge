#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>

namespace stereoforge::video {
// Caller owns buffers and orders upload -> preprocessing -> inference on stream.
void prepareGrayImage(const unsigned char* pixels, int source_width, int source_height,
    void* tensor, bool half, int width, int height, int canvas_width, int canvas_height,
    float* statistics, cudaStream_t stream);
void validateDescriptors(const void* descriptors, bool half, std::size_t count,
    float* statistics, cudaStream_t stream);
}  // namespace stereoforge::video
