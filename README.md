# Supro: An Advanced Motion Tracking System

![Supro Logo](1x/SuproLogo.gif)

A robust real-time motion tracking system using MediaPipe and OpenCV for tracking arm movements, hand gestures, and pose estimation. The system features advanced Kalman filtering, 3D visualization, and a Clay (for C++ integration)/ Qt-based (for Python integration) GUI interface. This README covers both C++ and Python implementations, with detailed sections for each.

> **Note**: The C++ implementation is currently configured for macOS systems with Apple Silicon. Compatibility with other platforms may require additional modifications to `CMakeLists.txt` and dependency paths.

---

## Table of Contents

1. [Introduction](#introduction)
2. [C++ Implementation](#c-implementation)
    - [Features](#features)
    - [Directory Structure](#directory-structure)
    - [Clay UI Integration](#clay-ui-integration)
    - [Installation & Build Instructions](#installation--build-instructions)
    - [Usage](#usage)
    - [Technical Details](#technical-details)
    - [Troubleshooting](#troubleshooting)
3. [Python Implementation](#python-implementation)
    - [Features](#features-1)
    - [Installation](#installation)
    - [Usage](#usage-1)
    - [Technical Details](#technical-details-1)
4. [Contributing](#contributing)
5. [License](#license)
6. [Acknowledgments](#acknowledgments)


---

## Introduction

The **Advanced Motion Tracking System** uses cutting-edge computer vision technologies to enable real-time tracking of arm movements and gestures. This project features two distinct implementations:

- **C++ Implementation**: Built with C++20, OpenCV, and MediaPipe, and integrates Python bindings via PyBind11. This is the main application and will be the one updated going forward. There is a large emphasis on real-time performance, building every aspect to be as fast as possible. 
- **Python Implementation**: A flexible and interactive system leveraging Python libraries such as MediaPipe, OpenCV, and Matplotlib. This was mainly made as a proof of concept.

---

## C++ Implementation

### Features 

- **Pose Tracking**: Uses MediaPipe’s pose detection to track arm and hand movements.
- **OpenCV Integration**: Provides camera input, image processing, and real-time visualization.
- **PyBind11 Embedding**: Embeds Python modules for additional functionality, such as MediaPipe pipelines.
- **Kalman Filtering**: Smoothens tracking data to minimize noise.
- **Clay UI Integration**: Leverages the lightweight "Clay" library for responsive and efficient UI layout.

---

### Directory Structure 

```
C_with_key_bindings/
├── CMakeLists.txt                # Top-level CMake configuration
├── requirements.txt              # Python dependencies
├── build/                        # Generated build artifacts
│   ├── arm_tracker               # Built executable
│   ├── fonts/                    # Fonts copied for rendering
│   └── CMakeFiles/               # CMake internal files
├── fonts/
│   └── Roboto-Regular.ttf        # Font file for text rendering
├── include/                      # Header files
│   ├── arm_tracker.hpp           # Main header for tracking logic
│   ├── kalman_filter.hpp         # Kalman filter implementation
│   ├── mediapipe_wrapper.hpp     # MediaPipe-related functionality
│   ├── ui_wrapper.hpp            # Abstract UI wrapper
│   ├── visualizer.hpp            # Visualization helpers
│   ├── clay.h                    # "Clay" UI integration
│   ├── clay_ui_wrapper.hpp       # Wrapper for clay UI
│   └── stb_truetype.h            # Font rendering support
└── src/                          # Source files
    ├── arm_tracker.cpp           # Core tracking logic
    ├── main.cpp                  # Entry point of the application
    ├── bindings.cpp              # PyBind11 Python bindings
    ├── clay_impl.cpp             # Implementation for clay UI
    └── clay_ui_wrapper.cpp       # Wrapper for clay UI
```

---

### Clay UI Integration 

"Clay" is a high-performance 2D UI layout library designed for responsive and efficient layouts. Major features include:

- **Microsecond Layout Performance**: Optimized for speed with static arena-based memory use.
- **Declarative Syntax**: Nested, React-like syntax for complex layouts.
- **Renderer Agnostic**: Outputs sorted rendering primitives, compatible with 3D engines or HTML.
- **No Dependencies**: Self-contained `clay.h` with zero external dependencies.

#### Licensing

"Clay" by Nic Barker is licensed under the zlib/libpng license:

- Free to use, modify, and redistribute, including for commercial purposes.
- Requires acknowledgment in the product documentation if redistributed.

For more details, visit the [Clay GitHub repository](https://github.com/nicbarker/clay).

---

### Installation & Build Instructions 

#### Prerequisites
- **macOS with Apple Silicon**: The C++ implementation has been tested and configured specifically for macOS systems running on Apple Silicon.
- **CMake >= 3.10**
- **C++20 compiler** (Apple Clang, GCC, or MSVC)
- **OpenCV**
- **Eigen3**
- **Freetype**
- **Python 3.12**
- **PyBind11**

#### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Juliorodrigo23/motion-tracking-system
   cd motion-tracking-system/C_with_key_bindings
   ```

2. **Install Python Dependencies**:
   ```bash
   python3.12 -m pip install -r requirements.txt
   ```

3. **Configure and Build**:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

4. **Run the Executable**:
   ```bash
   ./arm_tracker
   ```

---

### Usage

The application performs pose detection using MediaPipe and visualizes the results with OpenCV. Key bindings are defined in `main.cpp`:

- **ESC**: Quit the application
- **F**: Pause/Resume Finger tracking
- **L/R**: Toggle left/right arm tracking

---

### Technical Details 

#### Core Components

1. **Pose Detection**:
   - MediaPipe’s pose model identifies arm and hand landmarks.

2. **Kalman Filtering**:
   - Implements state estimation for smoother tracking.

3. **UI and Visualization**:
   - OpenCV renders video feed with augmented tracking overlays.
   - Fonts are handled using `stb_truetype.h` and Freetype.

4. **Python Embedding**:
   - PyBind11 integrates Python code for extended functionality.

#### Coordinate Systems

- Camera frame: Raw MediaPipe output.
- World frame: Aligned coordinate system for visualization.
- Arm frame: Local wrist-centered coordinate system.

---

### Troubleshooting 

- **Missing Python or PyBind11**: Verify Python paths in `CMakeLists.txt`.
- **OpenCV Linking Errors**: Confirm the correct version of OpenCV is installed.
- **Build Errors on macOS**: Ensure Homebrew paths are correctly configured in `CMakeLists.txt`.

---

## Python Implementation

![Python Implementation Demo](1x/DemoForPythonTracking.gif)
> **Note**: The Python is not as comprehensive. The tracking lags behind by a few seconds and the supination/pronation is flipped.

### Features 

- **Dual Arm Tracking**: Tracks both arms and recognizes gestures.
- **Hand Gesture Recognition**: Detects pronation, supination, and finger states.
- **3D Pose Estimation**: Calculates joint positions and orientations.
- **Advanced Filtering**: Kalman filtering smoothens tracking data.
- **Real-Time Visualization**:
  - 3D skeleton visualization
  - Rotation angle plots
  - Gesture confidence metrics
  - Augmented video feed

---

### Installation 

1. Clone the repository:
   ```bash
   git clone https://github.com/Juliorodrigo23/motion-tracking-system
   cd motion-tracking-system/Python_version
   ```

2. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### Usage 

Run the main Python application:

```bash
python main.py
```

Controls:
- **L**: Toggle left arm tracking
- **R**: Toggle right arm tracking
- **1/2**: Toggle finger tracking for left/right hands
- **Q**: Quit application

---

### Technical Details 

#### Key Components

1. **RobustArmTracker**: Coordinates MediaPipe, Kalman filtering, and gesture recognition.
2. **GestureRecognizer**: Detects finger states, wrist rotations, and confidence metrics.
3. **Visualizer**: Provides real-time visualization using Matplotlib and OpenCV.

#### Performance Optimization

- Efficient Kalman filtering minimizes noise.
- Real-time updates ensure smooth visualization.

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This projects license is yet to be determined.

---

## Acknowledgments

- MediaPipe team for pose and hand tracking models.
- OpenCV community for computer vision tools.
- Nic Barker for the "Clay" UI library.
- Contributors: Julio Contreras.
- Advisor: Jorge Ortiz.
