cmake_minimum_required(VERSION 3.22)
project(StereoForgeVideo VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil libswscale)
find_package(nlohmann_json 3.10 REQUIRED)

add_library(stereoforge_video src/video_extractor.cpp)
target_compile_features(stereoforge_video PUBLIC cxx_std_20)
set_target_properties(stereoforge_video PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(stereoforge_video PUBLIC include PRIVATE src)
target_link_libraries(stereoforge_video PRIVATE PkgConfig::FFMPEG nlohmann_json::nlohmann_json)
target_compile_options(stereoforge_video PRIVATE -Wall -Wextra -Wpedantic)

add_executable(stereoforge-extract-frames src/main.cpp)
target_link_libraries(stereoforge-extract-frames PRIVATE stereoforge_video nlohmann_json::nlohmann_json)
target_compile_options(stereoforge-extract-frames PRIVATE -Wall -Wextra -Wpedantic)
install(TARGETS stereoforge-extract-frames RUNTIME DESTINATION bin)
