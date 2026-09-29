cmake_minimum_required(VERSION 3.22)
project(SceneForgeVideo VERSION 1.0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil libswscale)
find_package(nlohmann_json 3.10 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(sceneforge_video src/video_extractor.cpp src/parallel_extractor.cpp src/cuda_resize.cu)
set_target_properties(sceneforge_video PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "75;80;86;89;90")
target_compile_features(sceneforge_video PUBLIC cxx_std_20)
set_target_properties(sceneforge_video PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(sceneforge_video PUBLIC include)
target_link_libraries(sceneforge_video PRIVATE PkgConfig::FFMPEG nlohmann_json::nlohmann_json Threads::Threads CUDA::cudart CUDA::cuda_driver)
# NVCC-generated host code uses GNU line markers. Keep pedantic warnings for
# ordinary C++, but do not apply them to NVCC's generated translation units.
target_compile_options(sceneforge_video PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>")

add_executable(sceneforge-extract-frames src/main.cpp)
target_link_libraries(sceneforge-extract-frames PRIVATE sceneforge_video nlohmann_json::nlohmann_json)
target_compile_options(sceneforge-extract-frames PRIVATE -Wall -Wextra -Wpedantic)
install(TARGETS sceneforge-extract-frames RUNTIME DESTINATION bin)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features2d highgui)
add_library(sceneforge_keyframes src/keyframe_selector.cpp src/learned_features.cpp src/tensor_runner.cpp src/keyframe_kernels.cu src/cuda_ransac.cu)
set_target_properties(sceneforge_keyframes PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "75;80;86;89;90")
find_path(TENSORRT_INCLUDE NvInferRuntime.h PATHS /opt/tensorrt/include REQUIRED)
find_library(TENSORRT_LIBRARY NAMES nvinfer libnvinfer.so.10 PATHS /opt/tensorrt/lib REQUIRED)
target_compile_features(sceneforge_keyframes PUBLIC cxx_std_20)
target_include_directories(sceneforge_keyframes PUBLIC include ${OpenCV_INCLUDE_DIRS})
# TensorRT headers call deprecated APIs internally. Treat vendor headers as
# system headers while retaining deprecation warnings in our own source.
target_include_directories(sceneforge_keyframes SYSTEM PUBLIC ${TENSORRT_INCLUDE})
target_link_libraries(sceneforge_keyframes PUBLIC ${OpenCV_LIBS} ${TENSORRT_LIBRARY} CUDA::cudart PRIVATE sceneforge_video PkgConfig::FFMPEG Threads::Threads nlohmann_json::nlohmann_json)
target_compile_options(sceneforge_keyframes PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>")
add_executable(sceneforge-select-keyframes src/keyframes_main.cpp src/keyframe_debug_view.cpp)
target_link_libraries(sceneforge-select-keyframes PRIVATE sceneforge_keyframes nlohmann_json::nlohmann_json)
target_compile_options(sceneforge-select-keyframes PRIVATE -Wall -Wextra -Wpedantic)
install(TARGETS sceneforge-select-keyframes RUNTIME DESTINATION bin)

add_executable(sceneforge-match-pairs src/pair_matches_main.cpp)
target_link_libraries(sceneforge-match-pairs PRIVATE sceneforge_keyframes nlohmann_json::nlohmann_json)
target_compile_options(sceneforge-match-pairs PRIVATE -Wall -Wextra -Wpedantic)
install(TARGETS sceneforge-match-pairs RUNTIME DESTINATION bin)
