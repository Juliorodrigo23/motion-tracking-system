cmake_minimum_required(VERSION 3.10)
project(arm_tracker)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/arm_tracker.cpp
    src/mediapipe_wrapper.cpp
)

# Create library
add_library(arm_tracker ${SOURCES})
target_link_libraries(arm_tracker
    PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    pybind11::embed  # Use embed instead of module for calling Python from C++
)