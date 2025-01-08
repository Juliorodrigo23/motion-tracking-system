cmake_minimum_required(VERSION 3.10)
project(arm_tracker)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(mediapipe REQUIRED)

# Add source files
add_executable(arm_tracker
    main.cpp
    arm_tracker.cpp
)

# Include directories
target_include_directories(arm_tracker PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${MEDIAPIPE_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(arm_tracker PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    mediapipe::framework
)