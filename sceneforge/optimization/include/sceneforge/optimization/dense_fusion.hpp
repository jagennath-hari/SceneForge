// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Jagennath Hari
//
// This file is part of SceneForge.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
namespace sceneforge::optimization {
struct DensePreview {
    std::vector<std::array<float,3>> points;
    std::vector<std::array<std::uint8_t,3>> colors;
    std::vector<std::int64_t> frames;
    std::size_t total = 0;
};
class DenseFusion final {
public:
    explicit DenseFusion(double voxel_size);
    void Add(const float* points, const std::uint8_t* colors, std::size_t count, std::int64_t frame);
    [[nodiscard]] DensePreview Write(const std::string& path, std::size_t preview_limit) const;
private:
    struct Key {
        std::int64_t x,y,z;
        bool operator==(const Key&) const = default;
    };
    struct Hash { std::size_t operator()(const Key& key) const noexcept; };
    struct Cell {
        std::array<double,3> position{}, color{};
        std::uint64_t count = 0;
        std::int64_t first_frame = 0;
    };
    double voxel_size_;
    std::unordered_map<Key,Cell,Hash> cells_;
};
}
