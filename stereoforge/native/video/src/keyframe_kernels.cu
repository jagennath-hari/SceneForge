#include "stereoforge/video/keyframe_kernels.cuh"
#include <cuda_fp16.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace stereoforge::video {
namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("Keyframe preprocessing: ") + cudaGetErrorString(error));
}
template <typename T> __device__ float readValue(T value) { return static_cast<float>(value); }
template <> __device__ float readValue(__half value) { return __half2float(value); }
template <typename T> __device__ T writeValue(float value) { return static_cast<T>(value); }
template <> __device__ __half writeValue(float value) { return __float2half_rn(value); }

// Warp lanes follow adjacent output columns. Area integration prevents aliasing
// during downsampling. No block barrier is needed: each lane owns its output and
// the read-only source has no inter-thread dependency or reusable stencil here.
template <typename T>
__global__ void resizeNormalize(const unsigned char* source, int source_width, int source_height,
    T* output, int width, int height, int canvas_width, int canvas_height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= canvas_width || y >= canvas_height) return;
    float rgb[3]{};
    if (x < width && y < height) {
        const float scale_x = static_cast<float>(source_width) / width;
        const float scale_y = static_cast<float>(source_height) / height;
        const float left = x * scale_x, right = (x + 1) * scale_x;
        const float top = y * scale_y, bottom = (y + 1) * scale_y;
        for (int sy = static_cast<int>(floorf(top)); sy < static_cast<int>(ceilf(bottom)) && sy < source_height; ++sy) {
            const float wy = fminf(bottom, sy + 1.0f) - fmaxf(top, static_cast<float>(sy));
            for (int sx = static_cast<int>(floorf(left)); sx < static_cast<int>(ceilf(right)) && sx < source_width; ++sx) {
                const float wx = fminf(right, sx + 1.0f) - fmaxf(left, static_cast<float>(sx));
                const std::size_t offset = (static_cast<std::size_t>(sy) * source_width + sx) * 3;
                for (int channel = 0; channel < 3; ++channel)
                    rgb[channel] += source[offset + 2 - channel] * wx * wy;
            }
        }
        for (int channel = 0; channel < 3; ++channel) rgb[channel] /= 255.0f * scale_x * scale_y;
    }
    for (int channel = 0; channel < 3; ++channel)
        output[(channel * canvas_height + y) * canvas_width + x] = writeValue<T>(rgb[channel]);
}
__device__ int reflect(int value, int size) {
    if (size <= 1) return 0;
    if (value < 0) return -value;
    if (value >= size) return max(0, 2 * size - value - 2);
    return value;
}

// A 32x8 block shares a haloed tile for the Laplacian stencil. Every thread
// participates in both barriers, including partial blocks. Eight warp partials
// are reduced by the first warp; only two atomics per block reach global memory.
template <typename T>
__global__ void sharpness(const T* image, int width, int height, int stride, int canvas_height, float* statistics) {
    __shared__ float tile[10][34];
    __shared__ float partial_sum[8];
    __shared__ float partial_square[8];
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int origin_x = blockIdx.x * 32, origin_y = blockIdx.y * 8;
    for (int index = tid; index < 340; index += 256) {
        const int tx = index % 34, ty = index / 34;
        const int sx = reflect(origin_x + tx - 1, width);
        const int sy = reflect(origin_y + ty - 1, height);
        const int offset = sy * stride + sx, plane = stride * canvas_height;
        tile[ty][tx] = 255.0f * (0.299f * readValue(image[offset]) +
            0.587f * readValue(image[plane + offset]) + 0.114f * readValue(image[2*plane + offset]));
    }
    __syncthreads();
    float sum = 0;
    if (origin_x + lane < width && origin_y + warp < height) {
        sum = tile[warp][lane + 1] + tile[warp + 2][lane + 1] +
              tile[warp + 1][lane] + tile[warp + 1][lane + 2] - 4 * tile[warp + 1][lane + 1];
    }
    float square = sum * sum;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
        square += __shfl_down_sync(0xffffffffu, square, offset);
    }
    if (lane == 0) { partial_sum[warp] = sum; partial_square[warp] = square; }
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? partial_sum[lane] : 0;
        square = lane < 8 ? partial_square[lane] : 0;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
            square += __shfl_down_sync(0xffffffffu, square, offset);
        }
        if (lane == 0) { atomicAdd(statistics, sum); atomicAdd(statistics + 1, square); }
    }
}
template <typename T>
__global__ void finiteDescriptors(const T* data, std::size_t count, float* statistics) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool invalid = index < count && !isfinite(readValue(data[index]));
    if (__any_sync(0xffffffffu, invalid) && threadIdx.x % 32 == 0) atomicExch(statistics + 2, 1.0f);
}
// One warp aggregates feature validity, rather than one atomic per point.
template <typename T>
__global__ void keypointStatistics(const T* points, const T* scores, int count,
    int width, int height, float threshold, float* statistics) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool invalid = false, usable = false;
    if (i < count) {
        const float x = readValue(points[2*i]), y = readValue(points[2*i+1]), score = readValue(scores[i]);
        invalid = !isfinite(x) || !isfinite(y) || !isfinite(score);
        usable = !invalid && x >= 4 && y >= 4 && x < width-4 && y < height-4 && score >= threshold;
    }
    const unsigned valid_mask = __ballot_sync(0xffffffffu, usable);
    const bool any_invalid = __any_sync(0xffffffffu, invalid);
    if (threadIdx.x % 32 == 0) {
        atomicAdd(statistics + 3, static_cast<float>(__popc(valid_mask)));
        if (any_invalid) atomicExch(statistics + 2, 1.0f);
    }
}
template <typename T>
void launch(const unsigned char* pixels, int sw, int sh, void* output, int width, int height,
    int cw, int ch, float* statistics, cudaStream_t stream) {
    const dim3 block(32, 8);
    resizeNormalize<<<dim3((cw + 31) / 32, (ch + 7) / 8), block, 0, stream>>>(
        pixels, sw, sh, static_cast<T*>(output), width, height, cw, ch);
    check(cudaGetLastError());
    sharpness<<<dim3((width + 31) / 32, (height + 7) / 8), block, 0, stream>>>(
        static_cast<T*>(output), width, height, cw, ch, statistics);
    check(cudaGetLastError());
}
}
void prepareRgbImage(const unsigned char* pixels, int sw, int sh, void* tensor, bool half,
    int width, int height, int cw, int ch, float* statistics, cudaStream_t stream) {
    if (!pixels || !tensor || !statistics || sw < width || sh < height || width < 1 || height < 1 || cw < width || ch < height)
        throw std::invalid_argument("Invalid preprocessing dimensions or pointers");
    check(cudaMemsetAsync(statistics, 0, 4 * sizeof(float), stream));
    if (half) launch<__half>(pixels, sw, sh, tensor, width, height, cw, ch, statistics, stream);
    else launch<float>(pixels, sw, sh, tensor, width, height, cw, ch, statistics, stream);
}
void validateDescriptors(const void* data, bool half, std::size_t count, float* statistics, cudaStream_t stream) {
    const unsigned blocks = static_cast<unsigned>((count + 255) / 256);
    if (half) finiteDescriptors<<<blocks, 256, 0, stream>>>(static_cast<const __half*>(data), count, statistics);
    else finiteDescriptors<<<blocks, 256, 0, stream>>>(static_cast<const float*>(data), count, statistics);
    check(cudaGetLastError());
}
void validateKeypoints(const void* points, const void* scores, bool half, int count,
    int width, int height, float threshold, float* statistics, cudaStream_t stream) {
    const unsigned blocks = static_cast<unsigned>((count + 255) / 256);
    if (half) keypointStatistics<<<blocks,256,0,stream>>>(static_cast<const __half*>(points),
        static_cast<const __half*>(scores),count,width,height,threshold,statistics);
    else keypointStatistics<<<blocks,256,0,stream>>>(static_cast<const float*>(points),
        static_cast<const float*>(scores),count,width,height,threshold,statistics);
    check(cudaGetLastError());
}
}  // namespace stereoforge::video
