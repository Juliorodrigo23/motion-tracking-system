#pragma once

// C++ Standard Library
#include <string>
#include <map>
#include <deque>
#include <memory>

// built by me
#include "kalman_filter.hpp"
#include "mediapipe_wrapper.hpp"  // Add this include for MediaPipeWrapper

// Third-party libraries
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

class ArmTracker {
public:
    // Structs should be declared before they're used
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

    // Public methods
    ArmTracker();
    ~ArmTracker();

    // Main processing method
    void processFrame(const cv::Mat& frame, TrackingResult& result);
    
    void toggleArm(const std::string& side);
    void toggleFingers(const std::string& side);

private:
    // MediaPipe wrapper
    std::unique_ptr<MediaPipeWrapper> mp_wrapper;

    // Processing method for landmarks
    void processFrameWithLandmarks(
        const cv::Mat& frame,
        const Eigen::MatrixXd& pose_landmarks,  // Nx4 matrix (x,y,z,visibility)
        const std::vector<Eigen::MatrixXd>& hand_landmarks,  // Vector of Nx3 matrices (x,y,z)
        TrackingResult& result
    );

    // Gesture recognition
    GestureState detectRotationGesture(
        const std::string& side,
        const HandState& hand,
        const std::map<std::string, JointState>& joints
    );
    
    // Processing methods for landmarks
    HandState processHandLandmarks(
        const Eigen::MatrixXd& landmarks,
        const std::string& side
    );
    
    void processPoseLandmarks(
        const Eigen::MatrixXd& landmarks,
        std::map<std::string, JointState>& joints
    );

    // Utility functions
    Eigen::Vector3d calculatePalmNormal(const HandState& hand);
    double calculateFingerExtension(const std::vector<Eigen::Vector3d>& points);
    
    // State tracking
    std::map<std::string, bool> activeArms;
    std::map<std::string, bool> activeFingers;
    std::map<std::string, std::deque<Eigen::Vector3d>> palmHistory;
    std::map<std::string, std::deque<double>> rotationHistory;

    // Constants
    static constexpr int HISTORY_SIZE = 10;
    static constexpr double CONFIDENCE_THRESHOLD = 0.6;
    static constexpr double GESTURE_ANGLE_THRESHOLD = 2.8; // radians
};