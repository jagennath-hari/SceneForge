#pragma once

#include "stereoforge/video/ffmpeg_resources.hpp"

namespace stereoforge::video::detail {

// Resize supported NVDEC surfaces in their owning CUDA context, then download.
// Other formats are downloaded unchanged for the CPU area resizer.
class CudaResizer final {
public:
    [[nodiscard]] FramePtr download(const AVFrame& source, int width, int height);
private:
    BufferPtr pool_;
};

}  // namespace stereoforge::video::detail
