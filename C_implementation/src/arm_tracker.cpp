// Project headers
#include "arm_tracker.hpp"

// Third-party libraries
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/landmark.pb.h>

struct ArmTracker::MediaPipeContext {
    mediapipe::CalculatorGraph graph;
    std::unique_ptr<mediapipe::OutputStreamPoller> posePoller;
    std::unique_ptr<mediapipe::OutputStreamPoller> handPoller;
};

ArmTracker::ArmTracker() : mpContext(std::make_unique<MediaPipeContext>()) {
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

    // Initialize MediaPipe graph
    std::string graphConfig = R"(
        input_stream: "input_video"
        output_stream: "pose_landmarks"
        output_stream: "hand_landmarks"
        
        node {
            calculator: "PoseLandmarkCpu"
            input_stream: "IMAGE:input_video"
            output_stream: "LANDMARKS:pose_landmarks"
            options {
                [mediapipe.PoseLandmarkCpuOptions.ext] {
                    min_detection_confidence: 0.5
                    min_tracking_confidence: 0.5
                    model_complexity: 1
                }
            }
        }
        node {
            calculator: "HandLandmarkCpu"
            input_stream: "IMAGE:input_video"
            output_stream: "LANDMARKS:hand_landmarks"
            options {
                [mediapipe.HandLandmarkOptions.ext] {
                    min_detection_confidence: 0.5
                    min_tracking_confidence: 0.5
                    max_num_hands: 2
                }
            }
        }
    )";

    mediapipe::CalculatorGraphConfig config;
    mediapipe::CalculatorGraphConfig::Parser().ParseFromString(graphConfig, &config);
    mpContext->graph.Initialize(config);

    // Set up output stream pollers
    mpContext->posePoller = std::make_unique<mediapipe::OutputStreamPoller>(
        mpContext->graph.AddOutputStreamPoller("pose_landmarks"));
    mpContext->handPoller = std::make_unique<mediapipe::OutputStreamPoller>(
        mpContext->graph.AddOutputStreamPoller("hand_landmarks"));

    mpContext->graph.StartRun({});
}

ArmTracker::~ArmTracker() {
    mpContext->graph.CloseAllPackets();
    mpContext->graph.WaitUntilDone();
}

void ArmTracker::processFrame(const cv::Mat& frame, TrackingResult& result) {
    // Prepare MediaPipe input
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, 
        frame.cols, frame.rows, 
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    
    cv::Mat rgb_frame;
    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
    rgb_frame.copyTo(cv::Mat(frame.rows, frame.cols, CV_8UC3, 
                            input_frame->MutablePixelData()));

    // Send frame to MediaPipe
    size_t frame_timestamp = 
        static_cast<size_t>(cv::getTickCount() * 1000000 / cv::getTickFrequency());
    mpContext->graph.AddPacketToInputStream(
        "input_video",
        mediapipe::Adopt(input_frame.release())
            .At(mediapipe::Timestamp(frame_timestamp)));

    // Get pose landmarks
    mediapipe::Packet packet;
    if (mpContext->posePoller->Next(&packet)) {
        auto& poseLandmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
        processJoints(poseLandmarks, result.joints);
        result.trackingLost = false;

        // Process hands if pose was detected
        if (mpContext->handPoller->Next(&packet)) {
            auto& handLandmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
            
            // Process each detected hand
            for (const auto& side : {"left", "right"}) {
                if (!activeArms[side]) continue;

                // Process hand landmarks if available
                HandState handState = processHandLandmarks(handLandmarks, side);
                if (handState.isTracked) {
                    result.hands[side] = handState;
                    // Calculate gesture only if we have both hand and joint data
                    if (result.joints.count(side + "_wrist")) {
                        result.gestures[side] = detectRotationGesture(
                            side, handState, result.joints);
                    }
                }
            }
        }
    } else {
        result.trackingLost = true;
    }
}

GestureState ArmTracker::detectRotationGesture(
    const std::string& side,
    const HandState& hand,
    const std::map<std::string, JointState>& joints) {
    
    if (!hand.isTracked || hand.landmarks.size() < 21) {
        return GestureState();
    }

    // Calculate palm normal
    Eigen::Vector3d palmNormal = calculatePalmNormal(hand);
    palmHistory[side].push_back(palmNormal);
    if (palmHistory[side].size() > HISTORY_SIZE) {
        palmHistory[side].pop_front();
    }

    // Calculate angle relative to vertical
    Eigen::Vector3d vertical(0, 1, 0);
    double angle = std::acos(palmNormal.dot(vertical));
    angle = angle * 180.0 / M_PI;  // Convert to degrees

    // Calculate palm rotation rate
    double palmRotationRate = 0.0;
    if (palmHistory[side].size() >= 2) {
        Eigen::Vector3d palmMovement = 
            palmHistory[side].back() - palmHistory[side][palmHistory[side].size() - 2];
        palmRotationRate = palmMovement.norm();
    }

    // Feature combination for robust detection
    double angleConfidence = std::abs(angle - 90.0) / 90.0;
    double rotationConfidence = std::min(palmRotationRate * 10.0, 1.0);

    // Combined confidence
    double confidence = std::min(1.0, (angleConfidence + rotationConfidence) / 2.0);

    // Adjust angle display for left hand
    double displayAngle = (side == "right") ? angle : (180.0 - angle);
    bool isPalmDown = (side == "right") ? (angle < 90) : (angle > 90);

    return GestureState(
        isPalmDown ? "pronation" : "supination",
        confidence,
        displayAngle
    );
}

Eigen::Vector3d ArmTracker::calculatePalmNormal(const HandState& hand) {
    // Get key points for palm plane
    const Eigen::Vector3d& wrist = hand.landmarks[0];  // Wrist
    const Eigen::Vector3d& mcp3 = hand.landmarks[9];   // Middle finger MCP
    const Eigen::Vector3d& mcp5 = hand.landmarks[17];  // Pinky MCP

    // Calculate palm vectors
    Eigen::Vector3d v1 = mcp3 - wrist;
    Eigen::Vector3d v2 = mcp5 - wrist;

    // Calculate normal using cross product
    Eigen::Vector3d normal = v1.cross(v2);
    return normal.normalized();
}

double ArmTracker::calculateFingerExtension(const std::vector<Eigen::Vector3d>& points) {
    if (points.size() < 3) return 0.0;

    // Calculate vectors between joints
    Eigen::Vector3d v1 = points[1] - points[0];
    Eigen::Vector3d v2 = points[2] - points[1];

    // Calculate angle between vectors
    double angle = std::acos(v1.dot(v2) / (v1.norm() * v2.norm()));
    
    return angle;
}

HandState ArmTracker::processHandLandmarks(
    const mediapipe::NormalizedLandmarkList& landmarks,
    const std::string& side) {
    
    HandState state;
    state.landmarks.reserve(landmarks.landmark_size());
    state.confidences.reserve(landmarks.landmark_size());

    for (const auto& landmark : landmarks.landmark()) {
        state.landmarks.push_back(Eigen::Vector3d(
            landmark.x(), landmark.y(), landmark.z()));
        state.confidences.push_back(landmark.visibility());
    }

    // Check if average confidence is above threshold
    double avgConfidence = 0;
    for (double conf : state.confidences) {
        avgConfidence += conf;
    }
    avgConfidence /= state.confidences.size();

    state.isTracked = avgConfidence > CONFIDENCE_THRESHOLD;
    return state;
}

void ArmTracker::processJoints(
    const mediapipe::NormalizedLandmarkList& poseLandmarks,
    std::map<std::string, JointState>& joints) {
    
    // Joint indices in MediaPipe pose landmarks
    const std::map<std::string, int> jointIndices = {
        {"left_shoulder", 11},
        {"left_elbow", 13},
        {"left_wrist", 15},
        {"right_shoulder", 12},
        {"right_elbow", 14},
        {"right_wrist", 16}
    };

    for (const auto& [jointName, index] : jointIndices) {
        const auto& landmark = poseLandmarks.landmark(index);
        
        JointState& joint = joints[jointName];
        Eigen::Vector3d measurement(landmark.x(), landmark.y(), landmark.z());
        
        // Apply Kalman filtering
        joint.kalman->predict();
        joint.kalman->update(measurement);
        
        // Use filtered position
        joint.position = joint.kalman->getPosition();
        joint.velocity = joint.kalman->getVelocity();
        joint.confidence = landmark.visibility();
    }
}

void ArmTracker::toggleArm(const std::string& side) {
    activeArms[side] = !activeArms[side];
}

void ArmTracker::toggleFingers(const std::string& side) {
    activeFingers[side] = !activeFingers[side];
}