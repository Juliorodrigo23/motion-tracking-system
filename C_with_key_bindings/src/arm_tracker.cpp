#include "arm_tracker.hpp"
#include <mediapipe/framework/calculator_framework.h>
#include <mediapipe/framework/formats/landmark.pb.h>

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

ArmTracker::~ArmTracker() {
    if (graph_.IsStarted()) {
        graph_.CloseAllPackets();
        graph_.WaitUntilDone();
    }
}

bool ArmTracker::initialize() {
    // Initialize MediaPipe graph
    const char* graphConfig = R"pb(
        input_stream: "input_video"
        output_stream: "pose_landmarks"
        output_stream: "hand_landmarks"
        
        node {
            calculator: "PoseLandmarkCpu"
            input_stream: "IMAGE:input_video"
            output_stream: "LANDMARKS:pose_landmarks"
            options {
                [mediapipe.PoseLandmarkCalculatorOptions.ext] {
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
    )pb";

    mediapipe::CalculatorGraphConfig config;
    if (!config.ParseFromString(graphConfig)) {
        return false;
    }

    auto status = graph_.Initialize(config);
    if (!status.ok()) {
        return false;
    }

    // Set up output stream pollers
    posePoller_ = std::make_unique<mediapipe::OutputStreamPoller>(
        graph_.AddOutputStreamPoller(kPoseOutputStream));
    handPoller_ = std::make_unique<mediapipe::OutputStreamPoller>(
        graph_.AddOutputStreamPoller(kHandOutputStream));

    status = graph_.StartRun({});
    return status.ok();
}

void ArmTracker::processFrame(const cv::Mat& frame, TrackingResult& result) {
    // Rest of your processFrame implementation remains largely the same,
    // but update to use the new member variables instead of mpContext
    auto input_frame = std::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, 
        frame.cols, frame.rows, 
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    
    cv::Mat rgb_frame;
    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
    rgb_frame.copyTo(cv::Mat(frame.rows, frame.cols, CV_8UC3, 
                            input_frame->MutablePixelData()));

    size_t frame_timestamp = 
        static_cast<size_t>(cv::getTickCount() * 1000000 / cv::getTickFrequency());
    
    auto status = graph_.AddPacketToInputStream(
        kInputStream,
        mediapipe::Adopt(input_frame.release())
            .At(mediapipe::Timestamp(frame_timestamp)));

    if (!status.ok()) {
        result.trackingLost = true;
        return;
    }

    // Get pose landmarks
    mediapipe::Packet packet;
    if (posePoller_->Next(&packet)) {
        auto& poseLandmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
        processJoints(poseLandmarks, result.joints);
        result.trackingLost = false;

        // Process hands if pose was detected
        if (handPoller_->Next(&packet)) {
            auto& handLandmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
            
            for (const auto& side : {"left", "right"}) {
                if (!activeArms[side]) continue;

                HandState handState = processHandLandmarks(handLandmarks, side);
                if (handState.isTracked) {
                    result.hands[side] = handState;
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

// Rest of your implementation methods can stay the same
// (detectRotationGesture, calculatePalmNormal, etc.)