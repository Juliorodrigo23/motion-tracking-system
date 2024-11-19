# Advanced Motion Tracking System

A robust real-time motion tracking system using MediaPipe and OpenCV for tracking arm movements, hand gestures, and pose estimation. The system features advanced Kalman filtering, 3D visualization, and a Qt-based GUI interface.

## Features

- **Dual Arm Tracking**: Simultaneous tracking of both left and right arms
- **Hand Gesture Recognition**: Detection of pronation/supination movements and finger states
- **3D Pose Estimation**: Real-time calculation of joint positions and orientations
- **Advanced Filtering**: Kalman filtering for smooth and stable tracking
- **Real-time Visualization**:
  - 3D arm position visualization
  - Rotation angle plots
  - Gesture confidence metrics
  - Live video feed with augmented tracking overlay
- **Interactive Controls**: Toggle tracking for individual arms and finger detection

## Requirements

- Python 3.8+
- OpenCV (cv2)
- MediaPipe
- NumPy
- FilterPy
- PySide6
- Matplotlib
- SciPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/juliorodrigo23/motion-tracking-system.git
cd motion-tracking-system
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main application:
```bash
python main.py
```

### Controls

- **L**: Toggle left arm tracking
- **R**: Toggle right arm tracking
- **1**: Toggle left hand finger tracking
- **2**: Toggle right hand finger tracking
- **Q**: Quit application

## System Architecture

### Key Components

1. **RobustArmTracker**: Main tracking system that coordinates:
   - MediaPipe pose and hand detection
   - Kalman filtering for joint tracking
   - Gesture recognition

2. **GestureRecognizer**: Processes hand landmarks to recognize:
   - Finger states (extended/flexed)
   - Wrist rotation (pronation/supination)
   - Gesture confidence metrics

3. **Visualizer**: Provides real-time visualization including:
   - 3D skeletal visualization
   - Rotation tracking graphs
   - Confidence metrics
   - Augmented video feed

4. **AdvancedKalmanFilter**: Implements advanced state estimation for:
   - Position tracking
   - Velocity estimation
   - Acceleration monitoring

## Technical Details

### Coordinate Systems

The system uses multiple coordinate frames:
- Camera frame: Raw MediaPipe coordinates
- World frame: Natural coordinate system (forward, up, right)
- Arm frame: Local coordinate system at wrist

### Tracking Pipeline

1. Frame Acquisition
2. Pose Detection
3. Hand Landmark Detection
4. Joint Filtering
5. Gesture Recognition
6. Visualization Update

### Performance Considerations

- Efficient Kalman filtering for smooth tracking
- Optimized visualization updates
- Configurable confidence thresholds

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MediaPipe team for their pose and hand tracking models
- OpenCV community for computer vision tools
- Qt team for the GUI framework
- Contributors: Julio Contreras, Jake Gianotto, Changmin Yu, Maahin Rathinagiriswaran
- Advisor: Jorge Ortiz

## Contact

Julio Contreras - [@juliooocon](https://www.linkedin.com/in/juliooocon/)
Project Link: [https://github.com/juliorodrigo23/motion-tracking-system](https://github.com/juliorodrigo23/motion-tracking-system)