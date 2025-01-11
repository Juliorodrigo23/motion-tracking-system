#pragma once
#include "arm_tracker.hpp"
#include <opencv2/opencv.hpp>
#include <mediapipe/framework/formats/landmark.pb.h>

class TrackerVisualizer {
public:
    TrackerVisualizer(int width, int height) 
        : frameWidth(width), frameHeight(height) {
        colors["left"] = cv::Scalar(0, 255, 0);   // Green
        colors["right"] = cv::Scalar(0, 0, 255);  // Red
        font = cv::FONT_HERSHEY_SIMPLEX;
    }
    
    void drawOverlay(cv::Mat& frame, const ArmTracker::TrackingResult& result);
    void drawPoseLandmarks(cv::Mat& frame, const mediapipe::NormalizedLandmarkList& landmarks);

private:
    cv::Point toPixel(const Eigen::Vector3d& pos);
    cv::Point2f normalizedToPixel(float x, float y) {
        return cv::Point2f(x * frameWidth, y * frameHeight);
    }
    void drawHand(cv::Mat& frame, const ArmTracker::HandState& hand, const cv::Scalar& color);
    void drawRotationInfo(cv::Mat& frame, const ArmTracker::GestureState& gesture,
                        const cv::Point& pos, const cv::Scalar& color,
                        const std::string& side);
    
    int frameWidth;
    int frameHeight;
    std::map<std::string, cv::Scalar> colors;
    int font;
};