cmake_minimum_required(VERSION 3.10)
project(arm_tracker)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
set(CMAKE_BUILD_TYPE Release)

# Add Homebrew prefix for M1 Mac
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    /opt/homebrew/include
)

# Source files
set(SOURCES
    src/arm_tracker.cpp
    src/main.cpp
)

# Add executable
add_executable(arm_tracker ${SOURCES})

# Link libraries
target_link_libraries(arm_tracker
    PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# macOS specific settings
if(APPLE)
    set(CMAKE_MACOSX_RPATH 1)
    target_link_options(arm_tracker PRIVATE -Wl,-dead_strip)
endif()