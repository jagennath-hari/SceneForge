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
target_link_libraries(stereoforge_optimization PUBLIC cunls::cunls nlohmann_json::nlohmann_json CUDA::cudart PRIVATE CUDA::cublas)
target_compile_options(stereoforge_optimization PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>")
add_executable(stereoforge-bundle-adjust src/main.cpp)
target_link_libraries(stereoforge-bundle-adjust PRIVATE stereoforge_optimization)
set_target_properties(stereoforge-bundle-adjust PROPERTIES INSTALL_RPATH /opt/cunls/lib)
install(TARGETS stereoforge-bundle-adjust RUNTIME DESTINATION bin)
