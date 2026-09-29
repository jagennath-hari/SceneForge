# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jagennath Hari
#
# This file is part of SceneForge.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.22)
project(SceneForgeOptimization LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(cunls CONFIG REQUIRED)
add_library(sceneforge_optimization src/factors.cu src/bundle_adjuster.cpp)
target_compile_features(sceneforge_optimization PUBLIC cxx_std_20)
set_target_properties(sceneforge_optimization PROPERTIES
    CXX_EXTENSIONS OFF CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "75;80;86;89;90")
target_include_directories(sceneforge_optimization PUBLIC include)
# CUDA 13 can place libcudacxx below cccl; installed cuNLS headers require it.
foreach(cuda_include IN LISTS CUDAToolkit_INCLUDE_DIRS)
    if(EXISTS "${cuda_include}/cccl")
        target_include_directories(sceneforge_optimization SYSTEM PUBLIC "${cuda_include}/cccl")
    endif()
endforeach()
target_link_libraries(sceneforge_optimization PUBLIC cunls::cunls CUDA::cudart PRIVATE CUDA::cublas)
target_compile_options(sceneforge_optimization PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>")

# In-process map builder: CPU Eigen geometry and CUDA/cuNLS BA, no solver files.
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core calib3d)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED)
set_target_properties(sceneforge_optimization PROPERTIES POSITION_INDEPENDENT_CODE ON)
pybind11_add_module(_sceneforge_map src/python_bindings.cpp src/map_builder.cpp src/map_bridge.cpp src/map_alignment.cpp src/dense_fusion.cpp src/rerun_recorder.cpp)
target_compile_features(_sceneforge_map PRIVATE cxx_std_20)
set_target_properties(_sceneforge_map PROPERTIES CXX_EXTENSIONS OFF INSTALL_RPATH /opt/cunls/lib)
target_link_libraries(_sceneforge_map PRIVATE sceneforge_optimization Eigen3::Eigen ${OpenCV_LIBS} ${CMAKE_DL_LIBS})
target_include_directories(_sceneforge_map SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
target_compile_options(_sceneforge_map PRIVATE -Wall -Wextra -Wpedantic)
install(TARGETS _sceneforge_map LIBRARY DESTINATION "${Python_SITEARCH}")
