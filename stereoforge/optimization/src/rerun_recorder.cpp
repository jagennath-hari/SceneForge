#include "stereoforge/optimization/rerun_recorder.hpp"
#include "stereoforge/optimization/map_builder.hpp"
#include <dlfcn.h>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace stereoforge::optimization {
RerunRecorder::RerunRecorder(const std::string& path) {
    this->library_ = dlopen("/usr/local/lib/libstereoforge_rerun.so", RTLD_NOW | RTLD_LOCAL);
    if (this->library_ == nullptr) { throw std::runtime_error(std::string("Load Rust Rerun adapter: ") + dlerror()); }
    try {
        const Open open = reinterpret_cast<Open>(dlsym(this->library_, "sf_rerun_open"));
        this->snapshot_ = reinterpret_cast<SnapshotCall>(dlsym(this->library_, "sf_rerun_snapshot"));
        this->event_ = reinterpret_cast<EventCall>(dlsym(this->library_, "sf_rerun_status"));
        this->image_ = reinterpret_cast<ImageCall>(dlsym(this->library_, "sf_rerun_image"));
        this->close_ = reinterpret_cast<Close>(dlsym(this->library_, "sf_rerun_close"));
        if (open == nullptr || this->snapshot_ == nullptr || this->close_ == nullptr || this->image_ == nullptr || this->event_ == nullptr) {
            throw std::runtime_error("Incompatible Rust Rerun adapter; rebuild Docker");
        }
        std::array<char,2048> error{};
        if (open(path.c_str(), &this->session_, error.data(), error.size()) != 0) {
            throw std::runtime_error(std::string("Open Rerun recording: ") + error.data());
        }
    } catch (...) { dlclose(this->library_); this->library_ = nullptr; throw; }
}
RerunRecorder::~RerunRecorder() {
    if (this->session_ != nullptr) { this->close_(this->session_); }
    // Rust/Rerun may own background threads or TLS destructors. Keep the DSO
    // loaded until process exit rather than unloading code those threads use.
}
void RerunRecorder::Event(const std::string& message) {
    std::array<char,2048> error{};
    if (this->event_(this->session_,message.c_str(),error.data(),error.size()) != 0) {
        throw std::runtime_error(error.data());
    }
}
void RerunRecorder::Image(const DepthFrame& frame) {
    if (frame.camera.width <= 0 || frame.camera.height <= 0 ||
        frame.rgb.size() != 3*static_cast<std::size_t>(frame.camera.width)*static_cast<std::size_t>(frame.camera.height)) {
        throw std::runtime_error("Invalid Rerun RGB dimensions");
    }
    std::array<char,2048> error{};
    if (this->image_(this->session_,frame.rgb.data(),frame.camera.width,frame.camera.height,
                     frame.id,error.data(),error.size()) != 0) { throw std::runtime_error(error.data()); }
}
void RerunRecorder::Snapshot(const SparseMap& map, std::uint32_t stage) {
    constexpr std::size_t point_limit = 50000;
    const std::size_t stride = std::max<std::size_t>(1, (map.landmarks.size()+point_limit-1)/point_limit);
    std::vector<float> xyz, cameras;
    std::vector<std::uint8_t> rgb;
    std::vector<std::int64_t> ids;
    xyz.reserve(3*std::min(point_limit,map.landmarks.size())); rgb.reserve(xyz.capacity());
    std::size_t index = 0;
    for (const std::pair<const TrackId,Landmark>& entry : map.landmarks) {
        if (index++ % stride != 0 || !entry.second.position.allFinite()) { continue; }
        for (int axis=0; axis<3; ++axis) { xyz.push_back(static_cast<float>(entry.second.position[axis])); }
        rgb.insert(rgb.end(),entry.second.color.begin(),entry.second.color.end());
    }
    cameras.reserve(18*map.cameras.size()); ids.reserve(map.cameras.size());
    for (const std::pair<const FrameId,Camera>& entry : map.cameras) {
        const Camera& camera = entry.second;
        ids.push_back(entry.first);
        for (int row=0; row<3; ++row) {
            for (int column=0; column<3; ++column) { cameras.push_back(static_cast<float>(camera.rotation(row,column))); }
        }
        for (int axis=0; axis<3; ++axis) { cameras.push_back(static_cast<float>(camera.center[axis])); }
        cameras.insert(cameras.end(), {static_cast<float>(camera.intrinsics(0,0)),static_cast<float>(camera.intrinsics(1,1)),
            static_cast<float>(camera.intrinsics(0,2)),static_cast<float>(camera.intrinsics(1,2)),
            static_cast<float>(camera.width),static_cast<float>(camera.height)});
    }
    std::array<char,2048> error{};
    if (this->snapshot_(this->session_,stage,xyz.data(),rgb.data(),xyz.size()/3,
                        cameras.data(),ids.data(),ids.size(),error.data(),error.size()) != 0) {
        throw std::runtime_error(std::string("Rerun snapshot: ") + error.data());
    }
}
} // namespace stereoforge::optimization
