#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace stereoforge::optimization {
struct SparseMap;
struct DepthFrame;
// Loads StereoForge's Rust SDK adapter, not the Rerun C++ SDK.
class RerunRecorder final {
public:
    explicit RerunRecorder(const std::string& path);
    ~RerunRecorder();
    RerunRecorder(const RerunRecorder&) = delete;
    RerunRecorder& operator=(const RerunRecorder&) = delete;
    void Event(const std::string& message);
    void Image(const DepthFrame& frame);
    void Dense(const float* xyz, const std::uint8_t* rgb, std::size_t count);
    void Snapshot(const SparseMap& map, std::uint32_t stage);
private:
    using Open = int (*)(const char*, void**, char*, std::size_t);
    using SnapshotCall = int (*)(void*, std::uint32_t, const float*, const std::uint8_t*, std::size_t,
                                 const float*, const std::int64_t*, std::size_t, char*, std::size_t);
    using ImageCall = int (*)(void*, const std::uint8_t*, std::uint32_t, std::uint32_t, std::int64_t, char*, std::size_t);
    using EventCall = int (*)(void*, const char*, char*, std::size_t);
    using Close = void (*)(void*);
    void* library_ = nullptr;
    void* session_ = nullptr;
    SnapshotCall snapshot_ = nullptr;
    EventCall event_ = nullptr;
    ImageCall image_ = nullptr;
    Close close_ = nullptr;
};
} // namespace stereoforge::optimization
