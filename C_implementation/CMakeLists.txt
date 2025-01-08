cmake_minimum_required(VERSION 3.10)
project(arm_tracker)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# macOS specific settings
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_BUILD_TYPE Release)

# Add MediaPipe
add_subdirectory(mediapipe)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Protobuf REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/mediapipe
    ${CMAKE_CURRENT_BINARY_DIR}
    /usr/local/include  # For Homebrew installed libraries
)

# Link directories
link_directories(
    /usr/local/lib  # For Homebrew installed libraries
)

# Source files
set(SOURCES
    src/arm_tracker.cpp
    src/main.cpp
    src/visualizer.cpp
)

# Add executable
add_executable(arm_tracker ${SOURCES})

# Link libraries
target_link_libraries(arm_tracker
    PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    mediapipe::framework
    ${Protobuf_LIBRARIES}
)

# macOS specific linking options
if(APPLE)
    target_link_options(arm_tracker PRIVATE -Wl,-dead_strip)
endif()