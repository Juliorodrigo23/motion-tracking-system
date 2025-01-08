#pragma once

// C++ Standard Library
#include <string>
#include <map>
#include <deque>
#include <memory>

// built by me
#include "kalman_filter.hpp"

// Third-party libraries
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

// Forward declarations for MediaPipe types
namespace mediapipe {
    class CalculatorGraph;
    class OutputStreamPoller;
    class NormalizedLandmarkList;
}

class ArmTracker {
public:
    struct GestureState {
        std::string type;  // "pronation" or "supination"
        double confidence;
        double angle;
        
        GestureState() : type("none"), confidence(0), angle(0) {}
        GestureState(const std::string& t, double c, double a) 
            : type(t), confidence(c), angle(a) {}
    };

    struct JointState {
        Eigen::Vector3d position;    // 3D position
        Eigen::Vector3d velocity;    // Velocity for smoothing
        double confidence;           // Detection confidence
        cv::Point2i pixelPos;       // 2D screen position
        std::unique_ptr<JointKalmanFilter> kalman;
    
        JointState() : confidence(0) {
            position = Eigen::Vector3d::Zero();
            velocity = Eigen::Vector3d::Zero();
            kalman = std::make_unique<JointKalmanFilter>();
        }
    };

    struct HandState {
        std::vector<Eigen::Vector3d> landmarks;
        std::vector<double> confidences;
        bool isTracked;
        
        HandState() : isTracked(false) {}
    };

    struct TrackingResult {
        bool trackingLost;
        std::map<std::string, JointState> joints;  // "shoulder", "elbow", "wrist"
        std::map<std::string, HandState> hands;    // "left", "right"
        std::map<std::string, GestureState> gestures; // "left", "right"
        
        TrackingResult() : trackingLost(true) {}
    };

    ArmTracker();
    ~ArmTracker();

    // Core tracking functions
    void processFrame(const cv::Mat& frame, TrackingResult& result);
    void toggleArm(const std::string& side);
    void toggleFingers(const std::string& side);

private:
    // Gesture recognition
    GestureState detectRotationGesture(const std::string& side,
                                    const HandState& hand,
                                    const std::map<std::string, JointState>& joints);
    
    // Hand processing
    HandState processHandLandmarks(const mediapipe::NormalizedLandmarkList& landmarks, 
                                const std::string& side);
    
    // Joint processing
    void processJoints(const mediapipe::NormalizedLandmarkList& poseLandmarks,
                    std::map<std::string, JointState>& joints);

    // Utility functions
    Eigen::Vector3d calculatePalmNormal(const HandState& hand);
    double calculateFingerExtension(const std::vector<Eigen::Vector3d>& points);
    
    // State tracking
    std::map<std::string, bool> activeArms;
    std::map<std::string, bool> activeFingers;
    std::map<std::string, std::deque<Eigen::Vector3d>> palmHistory;
    std::map<std::string, std::deque<double>> rotationHistory;
    
    // MediaPipe interface - implementation details hidden
    struct MediaPipeContext;
    std::unique_ptr<MediaPipeContext> mpContext;

    // Constants
    static constexpr int HISTORY_SIZE = 10;
    static constexpr double CONFIDENCE_THRESHOLD = 0.6;
    static constexpr double GESTURE_ANGLE_THRESHOLD = 2.8; // radians
};