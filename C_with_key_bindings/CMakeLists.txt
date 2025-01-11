cmake_minimum_required(VERSION 3.10)
project(arm_tracker)

# Set C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
set(CMAKE_BUILD_TYPE Release)

# Add Homebrew prefix for M1 Mac
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/lib/cmake")

# Specify exact Python paths
set(Python_ROOT_DIR "/Library/Frameworks/Python.framework/Versions/3.12")
set(Python_EXECUTABLE "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3")
set(Python_INCLUDE_DIRS "/Library/Frameworks/Python.framework/Versions/3.12/include/python3.12")
set(Python_LIBRARIES "/Library/Frameworks/Python.framework/Versions/3.12/lib/libpython3.12.dylib")
set(PYTHON_LIBRARY "/Library/Frameworks/Python.framework/Versions/3.12/lib/libpython3.12.dylib")
set(PYTHON_INCLUDE_DIR "/Library/Frameworks/Python.framework/Versions/3.12/include/python3.12")

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python 3.12 EXACT COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")
message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "PYTHONLIBS_VERSION_STRING: ${PYTHONLIBS_VERSION_STRING}")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/numpy/core/include"
)

# Source files
set(SOURCES
    src/arm_tracker.cpp
    src/main.cpp
)

# Create executable
add_executable(arm_tracker ${SOURCES})

# Link libraries
target_link_libraries(arm_tracker
    PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    pybind11::embed
    Python::Python
)

# macOS specific settings
if(APPLE)
    set(CMAKE_MACOSX_RPATH 1)
    target_link_options(arm_tracker PRIVATE -Wl,-dead_strip)
    
    # Add Python library path to rpath
    set_target_properties(arm_tracker PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "/Library/Frameworks/Python.framework/Versions/3.12/lib"
    )
endif()