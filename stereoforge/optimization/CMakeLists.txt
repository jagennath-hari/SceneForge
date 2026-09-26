cmake_minimum_required(VERSION 3.22)
project(StereoForgeOptimization LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(cunls CONFIG REQUIRED)
find_package(nlohmann_json 3.10 REQUIRED)
add_library(stereoforge_optimization src/factors.cu src/bundle_adjuster.cpp)
target_compile_features(stereoforge_optimization PUBLIC cxx_std_20)
set_target_properties(stereoforge_optimization PROPERTIES
    CXX_EXTENSIONS OFF CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "75;80;86;89;90")
target_include_directories(stereoforge_optimization PUBLIC include)
# CUDA 13 can place libcudacxx below cccl; installed cuNLS headers require it.
foreach(cuda_include IN LISTS CUDAToolkit_INCLUDE_DIRS)
    if(EXISTS "${cuda_include}/cccl")
        target_include_directories(stereoforge_optimization SYSTEM PUBLIC "${cuda_include}/cccl")
    endif()
endforeach()
target_link_libraries(stereoforge_optimization PUBLIC cunls::cunls CUDA::cudart PRIVATE CUDA::cublas)
target_compile_options(stereoforge_optimization PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>")
add_executable(stereoforge-bundle-adjust src/main.cpp)
target_link_libraries(stereoforge-bundle-adjust PRIVATE stereoforge_optimization nlohmann_json::nlohmann_json)
set_target_properties(stereoforge-bundle-adjust PROPERTIES INSTALL_RPATH /opt/cunls/lib)
install(TARGETS stereoforge-bundle-adjust RUNTIME DESTINATION bin)

# In-process map builder: CPU Eigen geometry and CUDA/cuNLS BA, no solver files.
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED)
set_target_properties(stereoforge_optimization PROPERTIES POSITION_INDEPENDENT_CODE ON)
pybind11_add_module(_stereoforge_map src/python_bindings.cpp src/map_builder.cpp src/map_alignment.cpp src/dense_fusion.cpp)
target_compile_features(_stereoforge_map PRIVATE cxx_std_20)
set_target_properties(_stereoforge_map PROPERTIES CXX_EXTENSIONS OFF INSTALL_RPATH /opt/cunls/lib)
target_link_libraries(_stereoforge_map PRIVATE stereoforge_optimization Eigen3::Eigen)
target_compile_options(_stereoforge_map PRIVATE -Wall -Wextra -Wpedantic)
install(TARGETS _stereoforge_map LIBRARY DESTINATION "${Python_SITEARCH}")
