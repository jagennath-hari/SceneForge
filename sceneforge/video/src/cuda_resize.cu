#include "sceneforge/video/cuda_resize.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
extern "C" {
#include <libavutil/hwcontext_cuda.h>
}
#include <cstdint>
#include <cmath>
#include <string>

namespace sceneforge::video::detail {
namespace {

class ContextGuard final {
public:
    explicit ContextGuard(CUcontext context) {
        if (cuCtxPushCurrent(context) != CUDA_SUCCESS)
            throw std::runtime_error("Cannot activate decoder CUDA context");
    }
    ~ContextGuard() { CUcontext previous{}; cuCtxPopCurrent(&previous); }
    ContextGuard(const ContextGuard&) = delete;
    ContextGuard& operator=(const ContextGuard&) = delete;
};

void cuda_check(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("CUDA frame resize: ") + cudaGetErrorString(result));
}

constexpr int warp_width = 32;
constexpr int block_rows = 4;
constexpr int block_threads = warp_width * block_rows;
// Bound per-block storage to leave room for multiple resident blocks. Actual
// occupancy and the best cutoff must be profiled on the target GPU.
constexpr std::size_t tile_budget_bytes = 16 * 1024;

// One warp owns adjacent scalar components in a row, including interleaved UV.
// Source loads are contiguous across lanes. Output pixels gather from the tile;
// no claim of conflict-free shared gathers is made for arbitrary resize ratios.
// The direct specialization handles footprints exceeding our shared-memory cap.
template<typename Sample, int Channels, int Shift, bool Tiled>
__global__ __launch_bounds__(block_threads) void resize_plane(
    const unsigned char* __restrict__ source, std::size_t source_pitch,
    unsigned char* __restrict__ destination, std::size_t destination_pitch,
    int source_width, int source_height, int width, int height,
    float scale_x, float scale_y, int tile_pitch) {
    extern __shared__ float tile[];
    const int scalar_x = blockIdx.x * warp_width + threadIdx.x;
    const int x = scalar_x / Channels;
    const int channel = scalar_x % Channels;
    const int y = blockIdx.y * block_rows + threadIdx.y;
    const int tile_x = static_cast<int>((blockIdx.x * (warp_width / Channels)) * scale_x);
    const int tile_y = static_cast<int>((blockIdx.y * block_rows) * scale_y);

    if constexpr (Tiled) {
        const int last_x = min(source_width, static_cast<int>(ceilf(
            min(static_cast<int>(blockIdx.x + 1) * (warp_width / Channels), width) * scale_x)));
        const int last_y = min(source_height, static_cast<int>(ceilf(
            min(static_cast<int>(blockIdx.y + 1) * block_rows, height) * scale_y)));
        const int columns = (last_x - tile_x) * Channels;
        const int rows = last_y - tile_y;
        // Each warp cooperatively loads rows, striding by a warp for wide tiles.
        // Different output warps consume these rows, requiring a block barrier.
        for (int row = threadIdx.y; row < rows; row += block_rows) {
            const Sample* input = reinterpret_cast<const Sample*>(source + (tile_y + row) * source_pitch);
            for (int column = threadIdx.x; column < columns; column += warp_width)
                tile[row * tile_pitch + column] = static_cast<float>(input[tile_x * Channels + column] >> Shift);
        }
        // No lane returns before this barrier, including partial edge blocks.
        __syncthreads();
    }
    if (x >= width || y >= height) return;

    const float left = x * scale_x;
    const float right = fminf((x + 1) * scale_x, static_cast<float>(source_width));
    const float top = y * scale_y;
    const float bottom = fminf((y + 1) * scale_y, static_cast<float>(source_height));
    const int first_x = static_cast<int>(left);
    const int last_x = min(source_width, static_cast<int>(ceilf(right)));
    const int first_y = static_cast<int>(top);
    const int last_y = min(source_height, static_cast<int>(ceilf(bottom)));
    float sum = 0.0f;
    for (int sy = first_y; sy < last_y; ++sy) {
        const float wy = fminf(bottom, sy + 1.0f) - fmaxf(top, static_cast<float>(sy));
        const Sample* input = reinterpret_cast<const Sample*>(source + sy * source_pitch);
        float horizontal = 0.0f;
        for (int sx = first_x; sx < last_x; ++sx) {
            const float wx = fminf(right, sx + 1.0f) - fmaxf(left, static_cast<float>(sx));
            float value;
            if constexpr (Tiled)
                value = tile[(sy - tile_y) * tile_pitch + (sx - tile_x) * Channels + channel];
            else
                value = static_cast<float>(input[sx * Channels + channel] >> Shift);
            horizontal = fmaf(value, wx, horizontal);
        }
        sum = fmaf(horizontal, wy, sum);
    }
    // Normalize once rather than accumulating and dividing weights per sample.
    // P010 is unpacked before integration and repacked only at the final store.
    const float area = (right - left) * (bottom - top);
    constexpr unsigned maximum = (1u << (sizeof(Sample) * 8 - Shift)) - 1u;
    const float value = fminf(static_cast<float>(maximum), fmaxf(0.0f, sum / area));
    Sample* output = reinterpret_cast<Sample*>(destination + y * destination_pitch);
    output[scalar_x] = static_cast<Sample>(static_cast<unsigned>(value + 0.5f) << Shift);
    // Tile storage is not reused, so no trailing barrier is needed.
}

template<typename Sample, int Channels, int Shift>
void launch_resize(const AVFrame& source, AVFrame& destination, int plane,
                   int source_width, int source_height, int width, int height, cudaStream_t stream) {
    const float scale_x = static_cast<float>(source_width) / width;
    const float scale_y = static_cast<float>(source_height) / height;
    // Two extra source pixels cover fractional tile boundaries and FP rounding.
    const std::size_t columns = (static_cast<std::size_t>(std::ceil((warp_width / Channels) * scale_x)) + 2) * Channels;
    const std::size_t rows = static_cast<std::size_t>(std::ceil(block_rows * scale_y)) + 2;
    const bool tiled = columns <= tile_budget_bytes / sizeof(float) &&
                       rows <= tile_budget_bytes / sizeof(float) / columns;
    const dim3 block(warp_width, block_rows);
    const dim3 grid((static_cast<unsigned>(width) * Channels + warp_width - 1) / warp_width,
                    (static_cast<unsigned>(height) + block_rows - 1) / block_rows);
    if (tiled)
        resize_plane<Sample, Channels, Shift, true><<<grid, block, columns * rows * sizeof(float), stream>>>(
            source.data[plane], source.linesize[plane], destination.data[plane], destination.linesize[plane],
            source_width, source_height, width, height, scale_x, scale_y, static_cast<int>(columns));
    else
        resize_plane<Sample, Channels, Shift, false><<<grid, block, 0, stream>>>(
            source.data[plane], source.linesize[plane], destination.data[plane], destination.linesize[plane],
            source_width, source_height, width, height, scale_x, scale_y, 0);
    cuda_check(cudaGetLastError());
}
}  // namespace

FramePtr CudaResizer::download(const AVFrame& source, int width, int height) {
    if (source.format != AV_PIX_FMT_CUDA || source.width <= 0 || source.height <= 0 ||
        width <= 0 || height <= 0 || width > source.width || height > source.height)
        throw std::invalid_argument("CUDA resize requires positive dimensions and no upscaling");
    if (!source.hw_frames_ctx) throw std::runtime_error("CUDA frame has no hardware context");
    const AVHWFramesContext* frames = reinterpret_cast<const AVHWFramesContext*>(source.hw_frames_ctx->data);
    const AVPixelFormat format = frames->sw_format;
    const bool supported = format == AV_PIX_FMT_NV12 || format == AV_PIX_FMT_P010LE || format == AV_PIX_FMT_YUV420P;
    FramePtr resized = require(FramePtr{av_frame_alloc()});
    const AVFrame* download = &source;
    if (supported && (source.width != width || source.height != height)) {
        const AVCUDADeviceContext* device = static_cast<const AVCUDADeviceContext*>(frames->device_ctx->hwctx);
        const ContextGuard context(device->cuda_ctx);
        const AVHWFramesContext* cached = this->pool_ ? reinterpret_cast<const AVHWFramesContext*>(this->pool_->data) : nullptr;
        if (!cached || cached->sw_format != format || cached->width != width || cached->height != height ||
            cached->device_ctx != frames->device_ctx) {
            this->pool_ = require(BufferPtr{av_hwframe_ctx_alloc(frames->device_ref)});
            AVHWFramesContext* target = reinterpret_cast<AVHWFramesContext*>(this->pool_->data);
            target->format = AV_PIX_FMT_CUDA;
            target->sw_format = format;
            target->width = width;
            target->height = height;
            check(av_hwframe_ctx_init(this->pool_.get()), "Create resized CUDA frame pool");
        }
        check(av_hwframe_get_buffer(this->pool_.get(), resized.get(), 0), "Allocate resized CUDA frame");
        const int planes = format == AV_PIX_FMT_YUV420P ? 3 : 2;
        for (int plane = 0; plane < planes; ++plane) {
            const int sw = plane == 0 ? source.width : (source.width + 1) / 2;
            const int sh = plane == 0 ? source.height : (source.height + 1) / 2;
            const int dw = plane == 0 ? width : (width + 1) / 2;
            const int dh = plane == 0 ? height : (height + 1) / 2;
            if (format == AV_PIX_FMT_P010LE) {
                if (plane == 0)
                    launch_resize<std::uint16_t, 1, 6>(source, *resized, plane, sw, sh, dw, dh, device->stream);
                else
                    launch_resize<std::uint16_t, 2, 6>(source, *resized, plane, sw, sh, dw, dh, device->stream);
            } else if (format == AV_PIX_FMT_NV12 && plane != 0)
                launch_resize<std::uint8_t, 2, 0>(source, *resized, plane, sw, sh, dw, dh, device->stream);
            else
                launch_resize<std::uint8_t, 1, 0>(source, *resized, plane, sw, sh, dw, dh, device->stream);
        }
        // One checked synchronization per frame, not per plane or per device.
        // Keep this error boundary: older FFmpeg CUDA transfer implementations
        // can log a failed CUDA copy/sync yet return success to their caller.
        cuda_check(cudaStreamSynchronize(device->stream));
        download = resized.get();
    }
    FramePtr result = require(FramePtr{av_frame_alloc()});
    check(av_hwframe_transfer_data(result.get(), download, 0), "Download decoded frame");
    check(av_frame_copy_props(result.get(), &source), "Copy frame color and timestamp metadata");
    return result;
}
}  // namespace sceneforge::video::detail
