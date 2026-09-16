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

// Keys cubic convolution, a=-0.5 (Catmull-Rom). Widening the support
// by the downsampling ratio supplies the low-pass filter for antialiasing.
__device__ float cubicWeight(float distance) {
    const float x = fabsf(distance);
    if (x < 1.0f) return ((1.5f*x - 2.5f)*x)*x + 1.0f;
    if (x < 2.0f) return ((-0.5f*x + 2.5f)*x - 4.0f)*x + 2.0f;
    return 0.0f;
}

// Each warp owns one source row and 32 adjacent output columns. Cooperatively
// load bounded 128-pixel strips, including the cubic halo. Chunking keeps shared
// memory bounded even for large reduction ratios. All 256 lanes reach both
// barriers on every iteration, including partial image blocks.
__global__ void bicubicHorizontal(const unsigned char* source, int source_width, int source_height,
    float* intermediate, int width) {
    __shared__ float tile[8][128];
    const int lane = threadIdx.x, row = threadIdx.y;
    const int origin_x = blockIdx.x*32, y = blockIdx.y*8 + row;
    const int x = origin_x + lane, channel = blockIdx.z;
    const float scale = static_cast<float>(source_width)/width;
    const float support = fmaxf(scale, 1.0f);
    const float center = (x+0.5f)*scale - 0.5f;
    const int first = static_cast<int>(ceilf((origin_x+0.5f)*scale - 0.5f - 2*support));
    const int last = static_cast<int>(floorf((min(origin_x+31,width-1)+0.5f)*scale - 0.5f + 2*support));
    const int tap_first = static_cast<int>(ceilf(center-2*support));
    const int tap_last = static_cast<int>(floorf(center+2*support));
    float sum = 0, weights = 0;
    for (int base = first; base <= last; base += 128) {
        for (int i = lane; i < 128; i += 32) {
            const int sx = min(max(base+i,0),source_width-1);
            tile[row][i] = y < source_height ? source[(static_cast<std::size_t>(y)*source_width+sx)*3+2-channel]/255.0f : 0;
        }
        __syncthreads();
        if (x < width && y < source_height) {
            for (int sx = max(base,tap_first); sx <= min(base+127,tap_last); ++sx) {
                const float weight = cubicWeight((sx-center)/support);
                sum = fmaf(weight, tile[row][sx-base], sum);
                weights += weight;
            }
        }
        __syncthreads();
    }
    if (x < width && y < source_height)
        intermediate[(static_cast<std::size_t>(channel)*source_height+y)*width+x] = sum/weights;
}

// Adjacent lanes load/store adjacent columns. Eight output rows reuse shared
// 32-row strips; the extra shared column avoids a power-of-two row pitch.
// Intermediate values remain FP32 and unclipped until the second pass so cubic
// negative lobes are retained. Only the final RGB values are clamped to [0,1].
template <typename T>
__global__ void bicubicVertical(const float* intermediate, int source_height, T* output,
    int width, int height, int canvas_width, int canvas_height) {
    __shared__ float tile[32][33];
    const int lane = threadIdx.x, row = threadIdx.y;
    const int x = blockIdx.x*32+lane, origin_y = blockIdx.y*8;
    const int y = origin_y+row, channel = blockIdx.z;
    const float scale = static_cast<float>(source_height)/height;
    const float support = fmaxf(scale,1.0f);
    const float center = (y+0.5f)*scale - 0.5f;
    const int first = static_cast<int>(ceilf((origin_y+0.5f)*scale - 0.5f - 2*support));
    const int last = static_cast<int>(floorf((min(origin_y+7,height-1)+0.5f)*scale - 0.5f + 2*support));
    const int tap_first = static_cast<int>(ceilf(center-2*support));
    const int tap_last = static_cast<int>(floorf(center+2*support));
    float sum = 0, weights = 0;
    for (int base = first; base <= last; base += 32) {
        for (int i = row; i < 32; i += 8) {
            const int sy = min(max(base+i,0),source_height-1);
            tile[i][lane] = x < width ? intermediate[(static_cast<std::size_t>(channel)*source_height+sy)*width+x] : 0;
        }
        __syncthreads();
        if (x < width && y < height) {
            for (int sy = max(base,tap_first); sy <= min(base+31,tap_last); ++sy) {
                const float weight = cubicWeight((sy-center)/support);
                sum = fmaf(weight,tile[sy-base][lane],sum);
                weights += weight;
            }
        }
        __syncthreads();
    }
    if (x < width && y < height)
        output[(static_cast<std::size_t>(channel)*canvas_height+y)*canvas_width+x] =
            writeValue<T>(fminf(1.0f,fmaxf(0.0f,sum/weights)));
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
void launch(const float* intermediate, int sh, void* output, int width, int height,
    int cw, int ch, float* statistics, cudaStream_t stream) {
    const dim3 block(32, 8);
    bicubicVertical<<<dim3((width + 31) / 32, (height + 7) / 8, 3), block, 0, stream>>>(
        intermediate, sh, static_cast<T*>(output), width, height, cw, ch);
    check(cudaGetLastError());
    sharpness<<<dim3((width + 31) / 32, (height + 7) / 8), block, 0, stream>>>(
        static_cast<T*>(output), width, height, cw, ch, statistics);
    check(cudaGetLastError());
}
}
void prepareRgbImage(const unsigned char* pixels, int sw, int sh, void* tensor, bool half,
    int width, int height, int cw, int ch, float* intermediate, std::size_t intermediate_bytes,
    float* statistics, cudaStream_t stream) {
    if (!pixels || !tensor || !statistics || !intermediate || sw < width || sh < height ||
        width < 1 || height < 1 || cw < width || ch < height ||
        sw > 65536 || sh > 65536 || cw > 4096 || ch > 4096)
        throw std::invalid_argument("Invalid preprocessing dimensions or pointers");
    const std::size_t required = static_cast<std::size_t>(width)*sh*3*sizeof(float);
    if (intermediate_bytes < required) throw std::invalid_argument("Bicubic workspace is too small");
    check(cudaMemsetAsync(statistics, 0, 4 * sizeof(float), stream));
    const std::size_t output_bytes = static_cast<std::size_t>(cw)*ch*3*(half ? sizeof(__half) : sizeof(float));
    check(cudaMemsetAsync(tensor,0,output_bytes,stream));  // Bottom/right canvas padding.
    bicubicHorizontal<<<dim3((width+31)/32,(sh+7)/8,3),dim3(32,8),0,stream>>>(pixels,sw,sh,intermediate,width);
    check(cudaGetLastError());
    if (half) launch<__half>(intermediate, sh, tensor, width, height, cw, ch, statistics, stream);
    else launch<float>(intermediate, sh, tensor, width, height, cw, ch, statistics, stream);
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
