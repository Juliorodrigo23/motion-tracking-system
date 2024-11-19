import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import threading
from queue import Queue
import time
from filterpy.kalman import KalmanFilter
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvasQTAgg
import sys

@dataclass
class FingerState:
    extended: bool
    tip_position: np.ndarray
    confidence: float

@dataclass
class GestureState:
    type: str  # 'pronation', 'supination'
    confidence: float
    angle: float  # rotation angle in degrees
    
@dataclass
class PoseState:
    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # 3x3 rotation matrix
    confidence: float

class AdvancedKalmanFilter(KalmanFilter):
    def __init__(self, dim_x=9, dim_z=3):
        super().__init__(dim_x=dim_x, dim_z=dim_z)
        self.initialize_matrices()
    
    def initialize_matrices(self):
        dt = 1/30.0
        # State: [x, y, z, vx, vy, vz, ax, ay, az]
        self.F = np.array([
            [1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],
            [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],
            [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],
            [0, 0, 0, 1, 0, 0, dt, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, dt, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, dt],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0]
        ])
        
        self.R *= 0.1  # Measurement noise
        self.Q *= 0.1  # Process noise
        self.P *= 1000  # Initial state covariance

    
class GestureRecognizer:
    def __init__(self, side, buffer_size=10):
        self.side = side
        self.thumb_positions = deque(maxlen=buffer_size)
        self.wrist_rotations = deque(maxlen=buffer_size)
        self.index_positions = deque(maxlen=buffer_size)
        self.palm_normals = deque(maxlen=buffer_size)
        self.finger_states = {
            'thumb': None,
            'index': None,
            'middle': None,
            'ring': None,
            'pinky': None
        }
        
    def _calculate_finger_state(self, landmarks, finger_name) -> FingerState:
        landmark_indices = {
            'thumb': [mp.solutions.hands.HandLandmark.THUMB_TIP, 
                     mp.solutions.hands.HandLandmark.THUMB_IP,
                     mp.solutions.hands.HandLandmark.THUMB_MCP],
            'index': [mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                     mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP,
                     mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP],
            'middle': [mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
                      mp.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP,
                      mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP],
            'ring': [mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
                    mp.solutions.hands.HandLandmark.RING_FINGER_DIP,
                    mp.solutions.hands.HandLandmark.RING_FINGER_PIP],
            'pinky': [mp.solutions.hands.HandLandmark.PINKY_TIP,
                     mp.solutions.hands.HandLandmark.PINKY_DIP,
                     mp.solutions.hands.HandLandmark.PINKY_PIP]
        }
        
        if finger_name not in landmark_indices:
            return None
            
        indices = landmark_indices[finger_name]
        points = []
        for idx in indices:
            landmark = landmarks.landmark[idx]
            points.append(np.array([landmark.x, landmark.y, landmark.z]))
            
        # Calculate finger extension using angles
        vectors = [points[i] - points[i+1] for i in range(len(points)-1)]
        angle = np.arccos(np.clip(np.dot(vectors[0], vectors[1]) / 
                                 (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])), -1.0, 1.0))
        
        # Consider finger extended if angle is greater than 160 degrees (2.8 radians)
        extended = angle > 2.8
        
        return FingerState(
            extended=extended,
            tip_position=points[0],
            confidence=landmarks.landmark[indices[0]].visibility
        )

    def update(self, hand_landmarks, pose_state: PoseState) -> GestureState:
        if not hand_landmarks:
            self.finger_states = {finger: None for finger in self.finger_states}
            return GestureState('pronation', 0.0, 0.0)
            
        # Update finger states
        for finger in self.finger_states:
            self.finger_states[finger] = self._calculate_finger_state(hand_landmarks, finger)
            
        thumb_tip = np.array([
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].z
        ])
        
        index_tip = np.array([
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].z
        ])
        
        wrist = np.array([
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].z
        ])
        
        middle_mcp = np.array([
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].x,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].y,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].z
        ])
        
        pinky_mcp = np.array([
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP].x,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP].y,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP].z
        ])
        
        palm_vector1 = middle_mcp - wrist
        palm_vector2 = pinky_mcp - wrist
        palm_normal = np.cross(palm_vector1, palm_vector2)
        palm_normal = palm_normal / np.linalg.norm(palm_normal)
        
        self.thumb_positions.append(thumb_tip - wrist)
        self.index_positions.append(index_tip - wrist)
        self.palm_normals.append(palm_normal)
        self.wrist_rotations.append(pose_state.rotation)
        
        return self._detect_rotation_gesture()
        
    def _detect_rotation_gesture(self) -> GestureState:
        if len(self.palm_normals) < 2:
            return GestureState('pronation', 0.0, 0.0)
        
        # Calculate palm orientation relative to vertical
        current_palm_normal = self.palm_normals[-1]
        vertical = np.array([0, 1, 0])
        
        # Calculate angle between palm normal and vertical
        angle = np.degrees(np.arccos(np.clip(np.dot(current_palm_normal, vertical), -1.0, 1.0)))
        
        # Calculate temporal features
        if len(self.palm_normals) >= 2:
            palm_movement = self.palm_normals[-1] - self.palm_normals[-2]
            palm_rotation_rate = np.linalg.norm(palm_movement)
        else:
            palm_rotation_rate = 0
            
        # Combine multiple features for robust detection
        thumb_height = self.thumb_positions[-1][1]  # Y component relative to wrist
        index_height = self.index_positions[-1][1]
        
        # Adjust palm down condition based on the side
        # For right hand: palm down when angle < 90
        # For left hand: palm down when angle > 90
        is_palm_down = angle < 90 if self.side == 'right' else angle > 90
        is_thumb_below = thumb_height < index_height
        
        # Calculate confidence based on multiple factors
        angle_confidence = abs(angle - 90) / 90.0  # How far from neutral position
        height_confidence = abs(thumb_height - index_height) / 0.1  # Normalized height difference
        rotation_confidence = min(palm_rotation_rate * 10, 1.0)  # Movement confidence
        
        # Combined confidence
        confidence = min(1.0, (angle_confidence + height_confidence + rotation_confidence) / 3)
        
        # For left hand, we need to adjust the angle to maintain consistency
        display_angle = angle if self.side == 'right' else 180 - angle
        
        if is_palm_down:
            return GestureState('pronation', confidence, display_angle)
        else:
            return GestureState('supination', confidence, display_angle)

class Visualizer:
    def __init__(self):
        self.debug_mode = True  
        self.rotation_history = {'left': deque(maxlen=100), 'right': deque(maxlen=100)}
        self.position_history = {'left': deque(maxlen=100), 'right': deque(maxlen=100)}

        self.fig = None
        self.ax_3d = None
        self.ax_rotation = None
        self.ax_confidence = None
        
    """def initialize_plots(self):
        self.fig = plt.figure(figsize=(15, 5))
        
        # Main plots
        self.ax_3d = self.fig.add_subplot(131, projection='3d')
        self.ax_3d.set_title('3D Arm Positions')
        
        self.ax_rotation = self.fig.add_subplot(132)
        self.ax_rotation.set_title('Wrist Rotation')
        
        self.ax_confidence = self.fig.add_subplot(133)
        self.ax_confidence.set_title('Gesture Confidence')
        
        # Common setup
        if isinstance(self.ax_3d, Axes3D):
            self.ax_3d.set_xlabel('X')
            self.ax_3d.set_ylabel('Y')
            self.ax_3d.set_zlabel('Z')
        
        plt.tight_layout()
        self.fig.show()
"""
    def update_visualizations(self, frame, tracking_data):
        try:
            if not self.fig:
                return
                
            # Clear previous plots
            self.ax_3d.cla()
            self.ax_rotation.cla()
            self.ax_confidence.cla()
            
            # Set titles (need to reset after clearing)
            self.ax_3d.set_title('3D Arm Positions')
            self.ax_rotation.set_title('Wrist Rotation')
            self.ax_confidence.set_title('Gesture Confidence')
            
            # Process each arm
            for side in ['left', 'right']:
                if side in tracking_data and tracking_data[side]:
                    arm_data = tracking_data[side]
                    self._update_3d_arm(arm_data, side)
                    if 'gesture' in arm_data:
                        self._update_rotation_plot(arm_data['gesture'], side)
                        self._update_confidence_plot(arm_data['gesture'], side)
            
            # Set labels for 3D plot
            if isinstance(self.ax_3d, Axes3D):
                self.ax_3d.set_xlabel('Right-Left')
                self.ax_3d.set_ylabel('Up-Down')
                self.ax_3d.set_zlabel('Forward-Back')
            
            # Adjust layout
            self.fig.tight_layout()
            
        except Exception as e:
            print(f"Error in visualization update: {e}")
            
    def _update_3d_arm(self, arm_data, side):
        """Update 3D visualization with fixed coordinate system"""
        self.ax_3d.set_title('3D Arm Positions')
        
        # Get joint positions
        positions = []
        for joint in ['shoulder', 'elbow', 'wrist']:
            if joint in arm_data and arm_data[joint] is not None:
                pos = arm_data[joint].position
                positions.append(pos)
        
        if len(positions) == 3:
            positions = np.array(positions)
            
            # Convert to visualization coordinates
            vis_positions = positions.copy()
            # Flip the X coordinate for the right side to match camera mirror
            if side == 'right':
                # X is left-right (positive right)
                vis_positions[:, 0] = positions[:, 2]
            else:
                # X is left-right (negative right)
                vis_positions[:, 0] = -positions[:, 2]
                
            # Y is up-down (same)
            vis_positions[:, 1] = positions[:, 1]
            # Z is forward-backward (negative forward)
            vis_positions[:, 2] = positions[:, 0]
            
            # Set colors based on side
            color = 'cyan' if side == 'left' else 'blue'
            
            # Upper arm
            self.ax_3d.plot([vis_positions[0, 0], vis_positions[1, 0]],
                        [vis_positions[0, 1], vis_positions[1, 1]],
                        [vis_positions[0, 2], vis_positions[1, 2]],
                        color=color, linewidth=3, label=f'{side.capitalize()} Upper Arm')
            
            # Forearm
            self.ax_3d.plot([vis_positions[1, 0], vis_positions[2, 0]],
                        [vis_positions[1, 1], vis_positions[2, 1]],
                        [vis_positions[1, 2], vis_positions[2, 2]],
                        color=color, linewidth=3, label=f'{side.capitalize()} Forearm')
            
            # Plot joints
            self.ax_3d.scatter(vis_positions[:, 0], vis_positions[:, 1], vis_positions[:, 2],
                        color=color, s=100)
            
            # Set consistent view limits
            limits = [-0.5, 0.5]
            self.ax_3d.set_xlim(limits)
            self.ax_3d.set_ylim(limits)
            self.ax_3d.set_zlim(limits)
            
            # Set labels and view angle
            self.ax_3d.set_xlabel('Right-Left')
            self.ax_3d.set_ylabel('Up-Down')
            self.ax_3d.set_zlabel('Forward-Back')
            
            # Set fixed viewpoint for stability
            self.ax_3d.view_init(elev=30, azim=45)
            
            # Add legend and grid
            #self.ax_3d.legend()
            self.ax_3d.grid(True)
    
    def _update_rotation_plot(self, gesture_state, side):
        """Update rotation plot for each arm"""
        if not gesture_state:
            return
            
        self.rotation_history[side].append(gesture_state.angle)
        x = np.arange(len(self.rotation_history[side]))
        y = np.array(list(self.rotation_history[side]))
        
        color = 'cyan' if side == 'left' else 'blue'
        self.ax_rotation.plot(x, y, color=color, label=f'{side.capitalize()} Arm')
        
        # Add reference lines
        self.ax_rotation.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        self.ax_rotation.axhline(y=180, color='g', linestyle='--', alpha=0.5)
        
        self.ax_rotation.set_ylim([-20, 200])
        self.ax_rotation.set_ylabel('Rotation Angle (degrees)')
        self.ax_rotation.set_xlabel('Time')
        self.ax_rotation.grid(True)
        self.ax_rotation.legend()
    
    def _update_confidence_plot(self, gesture_state, side):
        """Update confidence visualization"""
        if not gesture_state:
            return
            
        color = 'cyan' if side == 'left' else 'blue'
        x = 0 if side == 'left' else 1
        
        self.ax_confidence.bar(x, gesture_state.confidence, color=color,
                           alpha=0.6, width=0.4)
        
        # Add text annotations
        self.ax_confidence.text(x, gesture_state.confidence/2,
                            f"{gesture_state.type}\n{gesture_state.confidence:.2f}",
                            ha='center', va='center')
        
        self.ax_confidence.set_ylim([0, 1.2])
        self.ax_confidence.set_xticks([0, 1])
        self.ax_confidence.set_xticklabels(['Left Arm', 'Right Arm'])
        self.ax_confidence.grid(True, axis='y')
    
    def _update_debug_plots(self, tracking_data):
        """Update debug visualizations"""
        if not self.debug_mode:
            return
            
        for side in ['left', 'right']:
            if side in tracking_data and tracking_data[side]:
                arm_data = tracking_data[side]
                
                # Get positions for debug plots
                positions = []
                for joint in ['shoulder', 'elbow', 'wrist']:
                    if joint in arm_data and arm_data[joint] is not None:
                        pos = arm_data[joint].position
                        positions.append(pos)
                
                if len(positions) == 3:
                    positions = np.array(positions)
                    color = 'cyan' if side == 'left' else 'blue'
                    
                    # Plot raw positions
                    self._plot_arm_debug(self.ax_raw, positions, f'Raw {side.capitalize()}', color)
                    
                    # Plot transformed positions
                    if 'pose' in arm_data:
                        transformed_pos = self._transform_positions(positions, arm_data['pose'])
                        self._plot_arm_debug(self.ax_transformed, transformed_pos, 
                                         f'Transformed {side.capitalize()}', color)
                    
                    # Plot joint vectors
                    if len(positions) >= 2:
                        self._plot_joint_vectors(positions, side)
    
    def _plot_arm_debug(self, ax, positions, title_prefix, color):
        """Helper function to plot arm positions for debugging"""
        # Plot segments
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
               color=color, alpha=0.5)
        
        # Plot joints
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  color=color, s=50)
        
        # Add labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{title_prefix} Positions')
    
    def _plot_joint_vectors(self, positions, side):
        """Plot joint vectors for debugging"""
        color = 'cyan' if side == 'left' else 'blue'
        offset = -1 if side == 'left' else 1
        
        # Calculate vectors
        upper = positions[1] - positions[0]
        fore = positions[2] - positions[1]
        
        # Plot vectors
        self.ax_vectors.quiver(offset, 0, upper[0], upper[1],
                           angles='xy', scale_units='xy', scale=1,
                           color=color, alpha=0.7,
                           label=f'{side.capitalize()} Upper')
        self.ax_vectors.quiver(offset, 0, fore[0], fore[1],
                           angles='xy', scale_units='xy', scale=1,
                           color=color, alpha=0.3,
                           label=f'{side.capitalize()} Fore')
        
        self.ax_vectors.grid(True)
        self.ax_vectors.legend()
        self.ax_vectors.set_aspect('equal')
    
    def _transform_positions(self, positions, pose_state):
        """Transform positions using pose state"""
        transformed = positions.copy()
        for i in range(len(positions)):
            transformed[i] = pose_state.rotation.dot(positions[i] - positions[0])
        return transformed

class RobustArmTracker:
    def __init__(self, max_frames_queue=30, confidence_threshold=0.6):
        # Initialize MediaPipe components
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # Initialize queues
        self.frame_queue = Queue(maxsize=max_frames_queue)
        self.result_queue = Queue(maxsize=max_frames_queue)
        
        # Initialize thread control
        self.running = False
        self.processing_thread = None
        
        # Tracking state for both arms
        self.active_arms = {'left': True, 'right': True}
        self.active_fingers = {'left': True, 'right': True}

        # Separate Kalman filters for each arm
        self.joint_trackers = {
            'right_shoulder': AdvancedKalmanFilter(),
            'right_elbow': AdvancedKalmanFilter(),
            'right_wrist': AdvancedKalmanFilter(),
            'left_shoulder': AdvancedKalmanFilter(),
            'left_elbow': AdvancedKalmanFilter(),
            'left_wrist': AdvancedKalmanFilter()
        }
        
        # Initialize gesture recognizers with side information
        self.gesture_recognizers = {
            'right': GestureRecognizer('right'),
            'left': GestureRecognizer('left')
        }
        
        # Initialize visualizer
        self.visualizer = Visualizer()
        
        # Additional tracking parameters
        self.confidence_threshold = confidence_threshold
        self.last_positions = {}
        self.tracking_lost = False
        self.frames_since_detection = 0
        
        # Initialize windows
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Tracking View', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Original', 0, 0)
        cv2.moveWindow('Tracking View', 700, 0)
        
        # Frame dimensions
        self.frame_width = 640
        self.frame_height = 480

    def toggle_arm(self, side):
        """Toggle tracking for specified arm"""
        self.active_arms[side] = not self.active_arms[side]
        print(f"{side.capitalize()} arm tracking: {'ON' if self.active_arms[side] else 'OFF'}")
    
    def toggle_fingers(self, side):
        """Toggle finger tracking for specified side"""
        self.active_fingers[side] = not self.active_fingers[side]
        print(f"{side.capitalize()} finger tracking: {'ON' if self.active_fingers[side] else 'OFF'}")

    def _get_pose_state(self, tracking_data):
        """Calculate pose state from tracking data with fixed coordinate system"""
        if ('wrist' not in tracking_data or tracking_data['wrist'] is None or
            'elbow' not in tracking_data or tracking_data['elbow'] is None or
            'shoulder' not in tracking_data or tracking_data['shoulder'] is None):
            return PoseState(
                position=np.zeros(3),
                rotation=np.eye(3),
                confidence=0.0
            )
            
        try:
            # Get joint positions in MediaPipe coordinate system
            shoulder = np.array(tracking_data['shoulder'].position)
            elbow = np.array(tracking_data['elbow'].position)
            wrist = np.array(tracking_data['wrist'].position)
            
            # Convert from MediaPipe coordinates to a more natural coordinate system
            # MediaPipe uses: x right-left, y up-down, z forward-backward
            # We want: x forward, y up, z right
            for pos in [shoulder, elbow, wrist]:
                temp = pos.copy()
                pos[0] = -temp[2]  # Forward axis
                pos[2] = temp[0]   # Right axis
                # y stays the same (up axis)
            
            # Calculate arm vectors
            upper_arm = elbow - shoulder
            forearm = wrist - elbow
            
            # Normalize vectors
            upper_arm = upper_arm / (np.linalg.norm(upper_arm) + 1e-6)
            forearm = forearm / (np.linalg.norm(forearm) + 1e-6)
            
            # Define coordinate system at wrist
            z_axis = forearm  # Forward along forearm
            y_axis = np.cross(forearm, upper_arm)  # Up perpendicular to arm plane
            y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
            x_axis = np.cross(y_axis, z_axis)  # Complete right-handed system
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)
            
            # Build rotation matrix
            rotation = np.column_stack([x_axis, y_axis, z_axis])
            
            # Calculate confidence
            joint_confidence = (tracking_data['shoulder'].confidence +
                              tracking_data['elbow'].confidence +
                              tracking_data['wrist'].confidence) / 3.0
            
            return PoseState(
                position=wrist,
                rotation=rotation,
                confidence=joint_confidence
            )
            
        except Exception as e:
            print(f"Error in pose calculation: {e}")
            return PoseState(
                position=np.zeros(3),
                rotation=np.eye(3),
                confidence=0.0
            )
    
    def _process_frame(self, frame):
        """Improved frame processing with image dimensions"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tracking_view = frame.copy()
            h, w, _ = frame.shape
            
            # Initialize tracking data dictionary
            tracking_data = {'left': {}, 'right': {}}
            
            # Process with dimensions
            pose_results = self.pose.process(frame_rgb)
            hand_results = self.hands.process(frame_rgb)
            
            if pose_results.pose_landmarks:
                self.frames_since_detection = 0
                self.tracking_lost = False
                landmarks = pose_results.pose_landmarks.landmark
                
                # Store hand landmarks in tracking data
                if hand_results.multi_hand_landmarks:
                    for hand_idx, hand_landmark in enumerate(hand_results.multi_hand_landmarks):
                        handedness = hand_results.multi_handedness[hand_idx].classification[0]
                        side = handedness.label.lower()
                        if side in tracking_data:
                            tracking_data[side]['hand_landmarks'] = hand_landmark
                
                # Process both arms if active
                for side in ['left', 'right']:
                    if not self.active_arms[side]:
                        continue
                    
                    # Process joints with validation
                    side_tracking_data = {}
                    valid_tracking = True
                    
                    # Define landmarks for each arm
                    joint_landmarks = {
                        'shoulder': getattr(self.mp_pose.PoseLandmark, f'{side.upper()}_SHOULDER'),
                        'elbow': getattr(self.mp_pose.PoseLandmark, f'{side.upper()}_ELBOW'),
                        'wrist': getattr(self.mp_pose.PoseLandmark, f'{side.upper()}_WRIST')
                    }
                    
                    # Process each joint
                    for joint_name, landmark_idx in joint_landmarks.items():
                        landmark = landmarks[landmark_idx]
                        
                        # Validate landmark visibility
                        if landmark.visibility < self.confidence_threshold:
                            valid_tracking = False
                            break
                        
                        pos = np.array([landmark.x, landmark.y, landmark.z])
                        
                        # Apply Kalman filtering
                        joint_key = f"{side}_{joint_name}"
                        if joint_key not in self.last_positions:
                            self.joint_trackers[joint_key].x = np.array([*pos, 0, 0, 0, 0, 0, 0])
                            self.last_positions[joint_key] = pos
                        else:
                            self.joint_trackers[joint_key].predict()
                            self.joint_trackers[joint_key].update(pos)
                        
                        state = self.joint_trackers[joint_key].x
                        self.last_positions[joint_key] = pos
                        
                        # Store joint data
                        side_tracking_data[joint_name] = type('JointData', (), {
                            'position': state[:3],
                            'velocity': state[3:6],
                            'acceleration': state[6:],
                            'confidence': landmark.visibility,
                            'pixel_pos': (int(pos[0] * w), int(pos[1] * h))
                        })
                    
                    # Only process arm if all joints were tracked successfully
                    if valid_tracking:
                        # Calculate pose state
                        pose_state = self._get_pose_state(side_tracking_data)
                        side_tracking_data['pose'] = pose_state
                        
                        # Add hand landmarks if available
                        if side in tracking_data and 'hand_landmarks' in tracking_data[side]:
                            side_tracking_data['hand_landmarks'] = tracking_data[side]['hand_landmarks']
                        
                        # Update gesture
                        gesture_state = self.gesture_recognizers[side].update(
                            side_tracking_data.get('hand_landmarks'),
                            pose_state
                        )
                        
                        # Store all tracking data
                        side_tracking_data['gesture'] = gesture_state
                        tracking_data[side] = side_tracking_data  # Replace with complete data
                
                return {
                    'left': tracking_data['left'],
                    'right': tracking_data['right'],
                    'tracking_lost': False
                }
            
            # Handle tracking loss
            self.frames_since_detection += 1
            self.tracking_lost = True
            cv2.putText(tracking_view, "Tracking Lost", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Tracking View', tracking_view)
            
            return {
                'tracking_lost': True,
                'left': {},
                'right': {}
            }
        
        except Exception as e:
            print(f"Error in frame processing: {str(e)}")
            return {
                'tracking_lost': True,
                'left': {},
                'right': {}
            }

            
        
    def _draw_pose_landmarks(self, image, landmarks):
        """Draw all pose landmarks and connections"""
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            image,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
    def start(self):
        """Start the processing thread"""
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop(self):
        """Stop processing and release resources"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        self.pose.close()
        self.hands.close()
        cv2.destroyAllWindows()
    
    def add_frame(self, frame):
        """Add a new frame to the processing queue"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
    
    def get_latest_result(self):
        """Get the most recent tracking result"""
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None
    
    def _processing_loop(self):
        """Main processing loop running in separate thread"""
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                result = self._process_frame(frame)
                
                if not self.result_queue.full():
                    self.result_queue.put(result)
            else:
                time.sleep(0.001)  # Prevent CPU spinning

    def _draw_arm_segments(self, image, tracking_data):
        """Draw arm segments with toggleable finger tracking"""
        h, w = image.shape[:2]
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        for side, base_color in [('right', (0, 255, 255)), ('left', (0, 255, 0))]:
            if not self.active_arms[side] or side not in tracking_data:
                continue
            
            data = tracking_data[side]
            if not data:
                continue
            
            try:
                # Draw arm bones
                required_joints = ['shoulder', 'elbow', 'wrist']
                if not all(joint in data for joint in required_joints):
                    continue
                
                # Draw arm bones
                connections = [('shoulder', 'elbow'), ('elbow', 'wrist')]
                for start_joint, end_joint in connections:
                    start_pos = data[start_joint].pixel_pos
                    end_pos = data[end_joint].pixel_pos
                    
                    if all(isinstance(pos, tuple) and len(pos) == 2 for pos in [start_pos, end_pos]):
                        cv2.line(image, start_pos, end_pos, (0, 0, 0), 4)  # Shadow
                        cv2.line(image, start_pos, end_pos, base_color, 2)  # Main line

                # Draw hand landmarks if available and enabled
                if self.active_fingers[side] and 'hand_landmarks' in data:
                    mp_drawing.draw_landmarks(
                        image,
                        data['hand_landmarks'],
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(
                            color=base_color,
                            thickness=2,
                            circle_radius=2
                        ),
                        mp_drawing.DrawingSpec(
                            color=(255, 255, 255),
                            thickness=1,
                            circle_radius=1
                        )
                    )
                    
                # Draw joints
                for joint in required_joints:
                    if joint not in data:
                        continue
                        
                    joint_data = data[joint]
                    pos = joint_data.pixel_pos
                    
                    if not isinstance(pos, tuple) or len(pos) != 2:
                        continue
                    
                    cv2.circle(image, pos, 10, (0, 0, 0), -1)  # Shadow
                    cv2.circle(image, pos, 8, base_color, -1)  # Main circle
                    cv2.circle(image, pos, 10, (255, 255, 255), 1)  # White outline
                    
                    if hasattr(joint_data, 'velocity'):
                        velocity = joint_data.velocity[:2]
                        vel_magnitude = np.linalg.norm(velocity)
                        if vel_magnitude > 0.001:
                            vel_direction = velocity / vel_magnitude
                            end_point = (
                                int(pos[0] + vel_direction[0] * 50),
                                int(pos[1] + vel_direction[1] * 50)
                            )
                            cv2.arrowedLine(image, pos, end_point, (0, 0, 255), 2)
                    
                    label = f"{joint}"
                    label_pos = (pos[0] + 15, pos[1])
                    cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 0, 0), 3)
                    cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 255, 255), 1)

                # Draw coordinate system
                if 'pose' in data and data['pose'].confidence > 0:
                    wrist_pos = data['wrist'].pixel_pos
                    pose_state = data['pose']
                    axes_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
                    scale = 50
                    
                    for i, (axis, color) in enumerate(zip(pose_state.rotation.T, axes_colors)):
                        end_point = (
                            int(wrist_pos[0] + axis[0] * scale),
                            int(wrist_pos[1] + axis[1] * scale)
                        )
                        cv2.line(image, wrist_pos, end_point, (0, 0, 0), 3)
                        cv2.line(image, wrist_pos, end_point, color, 2)
                        label = ['X', 'Y', 'Z'][i]
                        cv2.putText(image, label, end_point, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)
                    
            except Exception as e:
                print(f"Error drawing {side} arm segments: {str(e)}")
                continue

    def _draw_tracking_status(self, image, tracking_data):
        """Draw tracking status overlay with 3D pose information"""
        h, w = image.shape[:2]
        
        # Create semi-transparent overlay for status
        overlay = image.copy()
        status_height = 60
        cv2.rectangle(overlay, (0, 0), (w, status_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        for side, base_color in [('left', (0, 255, 255)), ('right', (0, 255, 0))]:
            if not self.active_arms[side] or side not in tracking_data:
                continue
                
            data = tracking_data[side]
            confidence_x = w//2 if side == 'right' else 10
            
            # Draw tracking confidence bars
            if 'pose' in data:
                confidence = data['pose'].confidence
                cv2.putText(image, f"{side.upper()} Confidence:", 
                        (confidence_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1)
                
                bar_width = 100
                bar_height = 10
                bar_x = confidence_x
                bar_y = 30
                
                # Draw confidence bar background
                cv2.rectangle(image, (bar_x, bar_y), 
                            (bar_x + bar_width, bar_y + bar_height),
                            (100, 100, 100), -1)
                
                # Draw filled confidence bar
                filled_width = int(confidence * bar_width)
                cv2.rectangle(image, (bar_x, bar_y),
                            (bar_x + filled_width, bar_y + bar_height),
                            base_color, -1)
                
                # Draw confidence percentage
                cv2.putText(image, f"{confidence*100:.1f}%",
                        (bar_x + bar_width + 5, bar_y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_bone(self, image, tracking_data, joint1, joint2, color=(0, 255, 0)):
        """Draw connection between two joints with specified color"""
        if (joint1 in tracking_data and joint2 in tracking_data and
            isinstance(tracking_data[joint1], object) and isinstance(tracking_data[joint2], object)):
            
            try:
                pos1 = tracking_data[joint1].position
                pos2 = tracking_data[joint2].position
                
                start_point = (int(pos1[0] * image.shape[1]), int(pos1[1] * image.shape[0]))
                end_point = (int(pos2[0] * image.shape[1]), int(pos2[1] * image.shape[0]))
                
                cv2.line(image, start_point, end_point, color, 2)
            except Exception as e:
                print(f"Error drawing bone {joint1}-{joint2}: {e}")

    def _draw_gesture_info(self, image, tracking_data):
        """Draw gesture information overlay without finger state text"""
        h, w = image.shape[:2]
        
        for side, (base_color, y_offset) in [('left', ((0, 255, 255), h-150)), 
                                            ('right', ((0, 255, 0), h-80))]:
            if not self.active_arms[side] or side not in tracking_data:
                continue
                
            data = tracking_data[side]
            if 'gesture' not in data:
                continue
                
            gesture_state = data['gesture']
            
            overlay = image.copy()
            cv2.rectangle(overlay, (0, y_offset-60), (300, y_offset+20), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            
            # Draw gesture type
            cv2.putText(image, f"{side.upper()} - {gesture_state.type}",
                    (10, y_offset-30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, base_color, 2)
            
            # Draw confidence bar
            conf_width = int(200 * gesture_state.confidence)
            cv2.rectangle(image, (10, y_offset-20), (210, y_offset-10),
                        (100, 100, 100), -1)
            cv2.rectangle(image, (10, y_offset-20), (10 + conf_width, y_offset-10),
                        base_color, -1)

def main():
    # Initialize Qt application
    app = QApplication(sys.argv)
    
    # Create main window
    window = QMainWindow()
    window.setWindowTitle("Motion Tracker")
    window.setMinimumSize(1066, 800)
    
    # Create central widget and layout
    central = QWidget()
    window.setCentralWidget(central)
    layout = QVBoxLayout()
    
    # Create video widgets with increased size
    video_layout = QHBoxLayout()
    original_video = QLabel()
    tracking_video = QLabel()
    original_video.setMinimumSize(300, 200)
    tracking_video.setMinimumSize(300, 200)
    original_video.setAlignment(Qt.AlignCenter)
    tracking_video.setAlignment(Qt.AlignCenter)
    video_layout.addWidget(original_video)
    video_layout.addWidget(tracking_video)
    
    # Create plot widget
    plot_widget = QWidget()
    plot_widget.setMinimumHeight(300)
    plot_layout = QHBoxLayout()
    plot_widget.setLayout(plot_layout)
    
    # Create matplotlib figure with subplots
    plt.style.use('dark_background')  # Use dark theme for plots
    figure = plt.figure(figsize=(18, 4))
    gs = figure.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    
    # Create subplots
    ax_3d = figure.add_subplot(gs[0], projection='3d')
    ax_rotation = figure.add_subplot(gs[1])
    ax_confidence = figure.add_subplot(gs[2])
    
    # Set titles and labels
    ax_3d.set_title('3D Arm Positions', color='white')
    ax_rotation.set_title('Wrist Rotation', color='white')
    ax_confidence.set_title('Gesture Confidence', color='white')
    
    # Configure canvas
    canvas = FigureCanvasQTAgg(figure)
    plot_layout.addWidget(canvas)
    
    # Add layouts to main layout
    layout.addLayout(video_layout, stretch=2)
    layout.addWidget(plot_widget, stretch=1)
    central.setLayout(layout)
    
    # Set dark theme
    window.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #242424;
            color: white;
        }
        QLabel {
            border: 1px solid #00fff2;
            background-color: #000000;
        }
    """)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize tracker and set up visualizer
    tracker = RobustArmTracker()
    tracker.visualizer.fig = figure
    tracker.visualizer.ax_3d = ax_3d
    tracker.visualizer.ax_rotation = ax_rotation
    tracker.visualizer.ax_confidence = ax_confidence
    tracker.start()
    
    def update_frame():
        ret, frame = cap.read()
        if not ret:
            return
        
        frame = cv2.flip(frame, 1)
        tracker.add_frame(frame)
        result = tracker.get_latest_result()
        
        if result and not result.get('tracking_lost', True):
            # Create tracking view
            tracking_view = frame.copy()
            
            # Process both arms and draw overlays
            for side in ['left', 'right']:
                if tracker.active_arms[side] and side in result:
                    arm_data = result[side]
                    if all(joint in arm_data for joint in ['shoulder', 'elbow', 'wrist']):
                        tracker._draw_arm_segments(tracking_view, {side: arm_data})
            
            # Draw tracking status and gesture info
            tracker._draw_tracking_status(tracking_view, result)
            tracker._draw_gesture_info(tracking_view, result)
            
            # Convert frames to RGB for Qt
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tracking_view_rgb = cv2.cvtColor(tracking_view, cv2.COLOR_BGR2RGB)
            
            # Update video widgets
            h, w = frame_rgb.shape[:2]
            bytes_per_line = 3 * w
            
            for img, widget in [(frame_rgb, original_video), (tracking_view_rgb, tracking_video)]:
                q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                widget_size = widget.size()
                scaled_size = QSize(widget_size.width(), widget_size.height())
                q_img_scaled = q_img.scaled(scaled_size, Qt.KeepAspectRatio)
                widget.setPixmap(QPixmap.fromImage(q_img_scaled))
            
            # Update visualizations
            tracker.visualizer.update_visualizations(frame, result)
            canvas.draw()
            
        else:
            tracking_view = frame.copy()
            cv2.putText(tracking_view, "Tracking Lost", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            tracking_view_rgb = cv2.cvtColor(tracking_view, cv2.COLOR_BGR2RGB)
            h, w = tracking_view_rgb.shape[:2]
            bytes_per_line = 3 * w
            q_img = QImage(tracking_view_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            q_img_scaled = q_img.scaled(tracking_video.size(), Qt.KeepAspectRatio)
            tracking_video.setPixmap(QPixmap.fromImage(q_img_scaled))
    
    def handle_key(event):
        if event.key() == Qt.Key_L:
            tracker.toggle_arm('left')
        elif event.key() == Qt.Key_R:
            tracker.toggle_arm('right')
        elif event.key() == Qt.Key_1:
            tracker.toggle_fingers('left')
        elif event.key() == Qt.Key_2:
            tracker.toggle_fingers('right')
        elif event.key() == Qt.Key_Q:
            window.close()
    
    def cleanup():
        timer.stop()
        tracker.stop()
        cap.release()
    
    # Set up timer and event handling
    timer = QTimer()
    timer.timeout.connect(update_frame)
    window.keyPressEvent = handle_key
    window.closeEvent = lambda event: cleanup()
    
    # Start the timer
    timer.start(33)
    
    # Show window and run application
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()