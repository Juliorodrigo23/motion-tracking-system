#include "arm_tracker.hpp"
#include <Eigen/Geometry>

ArmTracker::ArmTracker() {
    // Initialize active tracking flags
    activeArms["left"] = true;
    activeArms["right"] = true;
    activeFingers["left"] = true;
    activeFingers["right"] = true;

    // Initialize history buffers
    palmHistory["left"] = std::deque<Eigen::Vector3d>(HISTORY_SIZE);
    palmHistory["right"] = std::deque<Eigen::Vector3d>(HISTORY_SIZE);
    rotationHistory["left"] = std::deque<double>(HISTORY_SIZE);
    rotationHistory["right"] = std::deque<double>(HISTORY_SIZE);
}

ArmTracker::~ArmTracker() = default;

void ArmTracker::processFrameWithLandmarks(
    const cv::Mat& frame,
    const Eigen::MatrixXd& pose_landmarks,
    const std::vector<Eigen::MatrixXd>& hand_landmarks,
    TrackingResult& result) {
    
    if (pose_landmarks.rows() == 0) {
        result.trackingLost = true;
        return;
    }

    // Process pose landmarks
    processPoseLandmarks(pose_landmarks, result.joints);
    result.trackingLost = false;

    // Process hand landmarks
    for (size_t i = 0; i < hand_landmarks.size(); ++i) {
        const auto& hand_lms = hand_landmarks[i];
        // Determine side based on wrist position relative to shoulders
        std::string side;
        if (hand_lms.rows() > 0) {
            // Get wrist position (first landmark)
            Eigen::Vector3d wrist = hand_lms.row(0).head<3>();
            
            // Get shoulder positions from pose landmarks
            Eigen::Vector3d left_shoulder = pose_landmarks.row(11).head<3>();  // MediaPipe pose indices
            Eigen::Vector3d right_shoulder = pose_landmarks.row(12).head<3>();
            
            // Compare distances to determine side
            double left_dist = (wrist - left_shoulder).norm();
            double right_dist = (wrist - right_shoulder).norm();
            side = (left_dist < right_dist) ? "left" : "right";
        } else {
            continue;
        }

        if (!activeArms[side]) continue;

        HandState handState = processHandLandmarks(hand_lms, side);
        if (handState.isTracked) {
            result.hands[side] = handState;
            if (result.joints.count(side + "_wrist")) {
                result.gestures[side] = detectRotationGesture(
                    side, handState, result.joints);
            }
        }
    }
}

void ArmTracker::processPoseLandmarks(
    const Eigen::MatrixXd& landmarks,
    std::map<std::string, JointState>& joints) {
    
    // MediaPipe Pose landmark indices
    const std::map<std::string, int> landmark_indices = {
        {"left_shoulder", 11},
        {"right_shoulder", 12},
        {"left_elbow", 13},
        {"right_elbow", 14},
        {"left_wrist", 15},
        {"right_wrist", 16}
    };

    for (const auto& [joint_name, index] : landmark_indices) {
        if (index >= landmarks.rows()) continue;

        auto& joint = joints[joint_name];
        
        // Get position and confidence
        joint.position = landmarks.row(index).head<3>();
        joint.confidence = landmarks(index, 3);  // Visibility score

        // Skip low confidence detections
        if (joint.confidence < CONFIDENCE_THRESHOLD) continue;

        // Update Kalman filter
        joint.kalman->predict();
        joint.kalman->update(joint.position);
        
        // Get smoothed position and velocity
        joint.position = joint.kalman->getPosition();
        joint.velocity = joint.kalman->getVelocity();
    }
}

HandState ArmTracker::processHandLandmarks(
    const Eigen::MatrixXd& landmarks,
    const std::string& side) {
    
    HandState state;
    if (landmarks.rows() < 21) {  // MediaPipe hand has 21 landmarks
        return state;
    }

    // Convert landmarks to 3D positions
    state.landmarks.reserve(landmarks.rows());
    state.confidences.reserve(landmarks.rows());
    
    for (int i = 0; i < landmarks.rows(); ++i) {
        state.landmarks.push_back(landmarks.row(i).head<3>());
        // Note: MediaPipe Python API doesn't provide per-landmark confidence,
        // so we use a default high confidence for detected landmarks
        state.confidences.push_back(1.0);
    }
    
    state.isTracked = true;
    return state;
}

GestureState ArmTracker::detectRotationGesture(
    const std::string& side,
    const HandState& hand,
    const std::map<std::string, JointState>& joints) {
    
    if (!hand.isTracked || hand.landmarks.size() < 21) {
        return GestureState();
    }

    // Calculate palm normal
    Eigen::Vector3d palm_normal = calculatePalmNormal(hand);
    
    // Update palm history
    palmHistory[side].push_front(palm_normal);
    palmHistory[side].pop_back();
    
    // Calculate rotation angle from palm history
    if (palmHistory[side].size() >= 2) {
        Eigen::Vector3d prev_normal = palmHistory[side][1];
        double angle = std::acos(palm_normal.dot(prev_normal));
        
        // Update rotation history
        rotationHistory[side].push_front(angle);
        rotationHistory[side].pop_back();
        
        // Calculate average rotation
        double avg_rotation = 0.0;
        for (double rot : rotationHistory[side]) {
            avg_rotation += rot;
        }
        avg_rotation /= rotationHistory[side].size();
        
        // Detect rotation type
        if (avg_rotation > GESTURE_ANGLE_THRESHOLD) {
            // Determine rotation type based on cross product
            Eigen::Vector3d cross = prev_normal.cross(palm_normal);
            std::string type = (cross.y() > 0) ? "pronation" : "supination";
            return GestureState(type, avg_rotation / GESTURE_ANGLE_THRESHOLD, avg_rotation);
        }
    }
    
    return GestureState();
}

Eigen::Vector3d ArmTracker::calculatePalmNormal(const HandState& hand) {
    // MediaPipe hand landmark indices for palm
    const int wrist_idx = 0;
    const int index_mcp_idx = 5;
    const int pinky_mcp_idx = 17;
    
    Eigen::Vector3d wrist = hand.landmarks[wrist_idx];
    Eigen::Vector3d index_mcp = hand.landmarks[index_mcp_idx];
    Eigen::Vector3d pinky_mcp = hand.landmarks[pinky_mcp_idx];
    
    // Calculate palm vectors
    Eigen::Vector3d palm_width = pinky_mcp - index_mcp;
    Eigen::Vector3d palm_length = (index_mcp + pinky_mcp) / 2 - wrist;
    
    // Calculate normal using cross product
    return palm_width.cross(palm_length).normalized();
}

double ArmTracker::calculateFingerExtension(const std::vector<Eigen::Vector3d>& points) {
    if (points.size() < 4) return 0.0;
    
    // Calculate joint angles
    Eigen::Vector3d v1 = points[1] - points[0];
    Eigen::Vector3d v2 = points[2] - points[1];
    Eigen::Vector3d v3 = points[3] - points[2];
    
    double angle1 = std::acos(v1.normalized().dot(v2.normalized()));
    double angle2 = std::acos(v2.normalized().dot(v3.normalized()));
    
    // Return average joint angle
    return (angle1 + angle2) / 2.0;
}

void ArmTracker::toggleArm(const std::string& side) {
    if (activeArms.count(side)) {
        activeArms[side] = !activeArms[side];
    }
}

void ArmTracker::toggleFingers(const std::string& side) {
    if (activeFingers.count(side)) {
        activeFingers[side] = !activeFingers[side];
    }
}