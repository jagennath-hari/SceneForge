#include "stereoforge/optimization/dense_fusion.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
namespace stereoforge::optimization {
DenseFusion::DenseFusion(double voxel_size) : voxel_size_(voxel_size) {
    if (!std::isfinite(voxel_size) || voxel_size <= 0) { throw std::invalid_argument("Invalid dense voxel size"); }
}
std::size_t DenseFusion::Hash::operator()(const Key& key) const noexcept {
    std::size_t seed = std::hash<std::int64_t>{}(key.x);
    seed ^= std::hash<std::int64_t>{}(key.y)+0x9e3779b9+(seed<<6)+(seed>>2);
    seed ^= std::hash<std::int64_t>{}(key.z)+0x9e3779b9+(seed<<6)+(seed>>2);
    return seed;
}
void DenseFusion::Add(const float* points, const std::uint8_t* colors, std::size_t count, std::int64_t frame) {
    for (std::size_t i=0; i<count; ++i) {
        std::array<std::int64_t,3> index;
        bool valid = true;
        for (int axis=0; axis<3; ++axis) {
            const double coordinate = std::floor(points[3*i+axis]/this->voxel_size_);
            if (!std::isfinite(coordinate) || std::abs(coordinate) >= 9e18) { valid=false; break; }
            index[axis] = static_cast<std::int64_t>(coordinate);
        }
        if (!valid) { continue; }
        const Key key{index[0],index[1],index[2]};
        if (!this->cells_.contains(key) && this->cells_.size() >= 5000000) {
            throw std::runtime_error("Dense fusion exceeded five million voxels; increase dense voxel size");
        }
        Cell& cell = this->cells_[key];
        cell.first_frame = cell.count == 0 ? frame : std::min(cell.first_frame,frame);
        ++cell.count;
        for (int axis=0; axis<3; ++axis) {
            cell.position[axis] += points[3*i+axis]; cell.color[axis] += colors[3*i+axis];
        }
    }
}
DensePreview DenseFusion::Write(const std::string& path, std::size_t preview_limit) const {
    static_assert(std::endian::native == std::endian::little);
    std::ofstream stream(path,std::ios::binary);
    stream.exceptions(std::ios::failbit|std::ios::badbit);
    stream << "ply\nformat binary_little_endian 1.0\nelement vertex " << this->cells_.size()
           << "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    DensePreview result; result.total=this->cells_.size();
    const std::size_t stride=std::max<std::size_t>(1,(this->cells_.size()+std::max<std::size_t>(1,preview_limit)-1)/std::max<std::size_t>(1,preview_limit));
    std::size_t index=0;
    for (const std::pair<const Key,Cell>& entry : this->cells_) {
        std::array<float,3> xyz; std::array<std::uint8_t,3> rgb;
        for (int axis=0; axis<3; ++axis) {
            xyz[axis]=static_cast<float>(entry.second.position[axis]/entry.second.count);
            rgb[axis]=static_cast<std::uint8_t>(std::clamp(std::lround(entry.second.color[axis]/entry.second.count),0L,255L));
        }
        stream.write(reinterpret_cast<const char*>(xyz.data()),3*sizeof(float));
        stream.write(reinterpret_cast<const char*>(rgb.data()),3);
        if (preview_limit && index++%stride==0) {
            result.points.push_back(xyz); result.colors.push_back(rgb); result.frames.push_back(entry.second.first_frame);
        }
    }
    stream.close(); return result;
}
}
