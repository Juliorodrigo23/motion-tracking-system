// arm_tracker.cpp
#include "arm_tracker.hpp"


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
    
    // Initialize MediaPipe wrapper
    mp_wrapper = std::make_unique<MediaPipeWrapper>();
    lastValidGestures["left"] = GestureState();
    lastValidGestures["right"] = GestureState();
}

ArmTracker::~ArmTracker() = default;

bool ArmTracker::processFrame(const cv::Mat& frame, TrackingResult& result, cv::Mat& debug_output) {
    try {
        Eigen::MatrixXd pose_landmarks;
        std::vector<Eigen::MatrixXd> hand_landmarks;
        
        // Always copy the input frame to debug_output initially
        frame.copyTo(debug_output);
        
        // Process frame - should always return true now
        if (mp_wrapper->process_frame(frame, pose_landmarks, hand_landmarks, debug_output,
                                    activeArms, activeFingers)) {
            
            if (pose_landmarks.rows() > 0) {
                processFrameWithLandmarks(frame, pose_landmarks, hand_landmarks, result);
                result.trackingLost = false;
            } else {
                result.trackingLost = true;
                result.joints.clear();
                result.hands.clear();
                result.gestures.clear();
            }
            
            return true;  // Always return true to keep program running
        }
        
        // This should never happen now, but just in case
        result.trackingLost = true;
        result.joints.clear();
        result.hands.clear();
        result.gestures.clear();
        return true;  // Keep program running
        
    } catch (const std::exception& e) {
        std::cerr << "Error in ArmTracker::processFrame: " << e.what() << std::endl;
        // Make sure we have a valid debug output
        if (debug_output.empty()) {
            frame.copyTo(debug_output);
        }
        // Clear all tracking data
        result.trackingLost = true;
        result.joints.clear();
        result.hands.clear();
        result.gestures.clear();
        return true;  // Keep program running
    }
}

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
                GestureState newGesture = detectRotationGesture(
                side, handState, result.joints);

                // Update only if we detect a valid gesture
                if (newGesture.type != "none") {
                lastValidGestures[side] = newGesture;
                }
            result.gestures[side] = lastValidGestures[side];
            }
        }
    }
}

ArmTracker::HandState ArmTracker::processHandLandmarks(
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

static constexpr double MIN_ROTATION_THRESHOLD = 0.05;  // Lowered from 0.15
static constexpr double ROTATION_SMOOTHING_FACTOR = 0.6; // Lowered from 0.7 for faster response
static constexpr int MIN_STABLE_FRAMES = 2;  // Lowered from 3
static constexpr double GESTURE_ANGLE_THRESHOLD = 0.1;  // Add if not defined elsewhere

ArmTracker::GestureState ArmTracker::detectRotationGesture(
    const std::string& side,
    const HandState& hand,
    const std::map<std::string, JointState>& joints) {
    
    if (!hand.isTracked || hand.landmarks.size() < 21) {
        return GestureState();
    }

    // Calculate palm normal
    Eigen::Vector3d palm_normal = calculatePalmNormal(hand);
    
    // Get forearm direction for reference
    std::string elbow_key = side + "_elbow";
    std::string wrist_key = side + "_wrist";
    
    if (joints.count(elbow_key) == 0 || joints.count(wrist_key) == 0) {
        return GestureState();
    }
    
    // Calculate forearm direction
    Eigen::Vector3d forearm_dir = (joints.at(wrist_key).position - 
                                joints.at(elbow_key).position).normalized();
                                
    // Calculate rotation axis and angle relative to anatomical reference
    Eigen::Vector3d rotation_axis = palm_normal.cross(forearm_dir);
    double rotation_angle = std::acos(std::clamp(palm_normal.dot(forearm_dir), -1.0, 1.0));
    
    // Debug output palm normal
    std::cout << "Palm normal " << side << ": " << palm_normal.transpose() << std::endl;
    std::cout << "Forearm direction: " << forearm_dir.transpose() << std::endl;
    std::cout << "Rotation axis: " << rotation_axis.transpose() << std::endl;
    std::cout << "Initial rotation angle: " << rotation_angle << std::endl;
    
    // Update palm history with anatomically aware normal
    palmHistory[side].push_front(palm_normal);
    if (palmHistory[side].size() > HISTORY_SIZE) {
        palmHistory[side].pop_back();
    }
    
    // Need at least a few frames for stable detection
    if (palmHistory[side].size() < MIN_STABLE_FRAMES) {
        std::cout << "Not enough frames for " << side << std::endl;
        return GestureState();
    }

    // Calculate smoothed rotation angle from palm history
    double cumulative_angle = 0.0;
    Eigen::Vector3d cumulative_axis = Eigen::Vector3d::Zero();
    int valid_samples = 0;

    for (size_t i = 1; i < palmHistory[side].size(); ++i) {
        Eigen::Vector3d curr_normal = palmHistory[side][i-1];
        Eigen::Vector3d prev_normal = palmHistory[side][i];
        
        // Calculate rotation angle
        double angle = std::acos(std::clamp(curr_normal.dot(prev_normal), -1.0, 1.0));
        
        std::cout << "Frame " << i << " angle: " << angle << " (threshold: " << MIN_ROTATION_THRESHOLD << ")" << std::endl;
        
        // Only count significant rotations
        if (angle > MIN_ROTATION_THRESHOLD) {
            cumulative_angle += angle;
            cumulative_axis += curr_normal.cross(prev_normal);
            valid_samples++;
            std::cout << "Valid rotation detected in frame " << i << std::endl;
        }
    }

    // If we don't have enough valid samples, no significant rotation
    if (valid_samples < MIN_STABLE_FRAMES - 1) {
        std::cout << "Not enough valid samples for " << side << " (" << valid_samples << " < " << (MIN_STABLE_FRAMES - 1) << ")" << std::endl;
        return GestureState();
    }

    double avg_angle = cumulative_angle / valid_samples;
    Eigen::Vector3d avg_axis = cumulative_axis.normalized();

    std::cout << "Average angle: " << avg_angle << ", Axis: " << avg_axis.transpose() << std::endl;

    // Apply exponential smoothing to rotation history
    rotationHistory[side].push_front(avg_angle);
    if (rotationHistory[side].size() > HISTORY_SIZE) {
        rotationHistory[side].pop_back();
    }

    // Calculate smoothed rotation
    double smoothed_rotation = 0.0;
    double weight_sum = 0.0;
    double weight = 1.0;

    for (double rot : rotationHistory[side]) {
        smoothed_rotation += rot * weight;
        weight_sum += weight;
        weight *= ROTATION_SMOOTHING_FACTOR;
    }
    smoothed_rotation /= weight_sum;

    std::cout << "Smoothed rotation: " << smoothed_rotation << " (threshold: " << GESTURE_ANGLE_THRESHOLD << ")" << std::endl;

    // Only detect rotation if it's significant
    if (smoothed_rotation > GESTURE_ANGLE_THRESHOLD) {
        bool is_supination;
        if (side == "left") {
            // For left arm, positive rotation around forearm axis is supination
            is_supination = rotation_axis.dot(Eigen::Vector3d::UnitY()) < 0;
            std::cout << "Left hand rotation axis Y: " << rotation_axis.dot(Eigen::Vector3d::UnitY()) 
                    << ", is_supination: " << is_supination << std::endl;
        } else {
            // For right arm, negative rotation around forearm axis is supination
            is_supination = rotation_axis.dot(Eigen::Vector3d::UnitY()) < 0;
            std::cout << "Right hand rotation axis Y: " << rotation_axis.dot(Eigen::Vector3d::UnitY()) 
                    << ", is_supination: " << is_supination << std::endl;
        }
        
        std::string type = is_supination ? "supination" : "pronation";
        double confidence = std::min(1.0, smoothed_rotation / (GESTURE_ANGLE_THRESHOLD * 2));
        
        std::cout << "Detected " << type << " with confidence " << confidence << std::endl;
        
        return GestureState(type, confidence, smoothed_rotation);
    }

    return GestureState();
}

Eigen::Vector3d ArmTracker::calculatePalmNormal(const HandState& hand) {
    // MediaPipe hand landmark indices
    const int WRIST = 0;
    const int THUMB_CMC = 1;
    const int INDEX_MCP = 5;
    const int MIDDLE_MCP = 9;
    const int RING_MCP = 13;
    const int PINKY_MCP = 17;
    const int MIDDLE_PIP = 10;
    const int MIDDLE_TIP = 12;

    // Get key points
    Eigen::Vector3d wrist = hand.landmarks[WRIST];
    Eigen::Vector3d thumb_cmc = hand.landmarks[THUMB_CMC];
    Eigen::Vector3d index_mcp = hand.landmarks[INDEX_MCP];
    Eigen::Vector3d middle_mcp = hand.landmarks[MIDDLE_MCP];
    Eigen::Vector3d ring_mcp = hand.landmarks[RING_MCP];
    Eigen::Vector3d pinky_mcp = hand.landmarks[PINKY_MCP];
    Eigen::Vector3d middle_pip = hand.landmarks[MIDDLE_PIP];
    Eigen::Vector3d middle_tip = hand.landmarks[MIDDLE_TIP];

    // Calculate robust palm direction vectors
    Eigen::Vector3d palm_center = (index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 4.0;
    Eigen::Vector3d palm_direction = (palm_center - wrist).normalized();
    
    // Calculate palm width vector (perpendicular to thumb-pinky line)
    Eigen::Vector3d thumb_pinky = (pinky_mcp - thumb_cmc).normalized();
    
    // Calculate finger direction (using middle finger as reference)
    Eigen::Vector3d finger_direction = (middle_tip - middle_mcp).normalized();
    
    // Calculate palm normal using multiple reference vectors
    Eigen::Vector3d normal1 = thumb_pinky.cross(palm_direction);
    Eigen::Vector3d normal2 = thumb_pinky.cross(finger_direction);
    
    // Combine normals with weights
    Eigen::Vector3d weighted_normal = (normal1 + normal2).normalized();
    
    // Debug output
    std::cout << "Palm center: " << palm_center.transpose() << std::endl;
    std::cout << "Palm direction: " << palm_direction.transpose() << std::endl;
    std::cout << "Thumb-pinky vector: " << thumb_pinky.transpose() << std::endl;
    std::cout << "Finger direction: " << finger_direction.transpose() << std::endl;
    std::cout << "Weighted normal: " << weighted_normal.transpose() << std::endl;
    
    return weighted_normal;
}



void ArmTracker::toggleArm(const std::string& side) {
    if (activeArms.count(side)) {
        activeArms[side] = !activeArms[side];
        // Fingers will be shown/hidden based on arm state in the drawing logic
        // but maintain their own toggle state
    }
}

void ArmTracker::toggleFingers(const std::string& side) {
    if (activeFingers.count(side)) {
        activeFingers[side] = !activeFingers[side];
    }
}