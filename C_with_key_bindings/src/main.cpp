#include "arm_tracker.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

// Mouse callback function for OpenCV window
void onMouse(int event, int x, int y, int flags, void* userdata) {
    // Handle mouse events if needed
}

int main() {
    std::cout << "Opening camera..." << std::endl;
    
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }
    
    int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    std::cout << "Camera opened. Resolution: " << frameWidth << "x" << frameHeight << std::endl;
    
    // Create window and set mouse callback
    cv::namedWindow("Tracking Debug");
    cv::setMouseCallback("Tracking Debug", onMouse);
    
    // Create tracker instance
    std::cout << "Initializing ArmTracker..." << std::endl;
    ArmTracker tracker;
    std::cout << "ArmTracker initialized" << std::endl;
    
    cv::Mat frame;
    int frame_count = 0;
    
    std::cout << "Starting main loop..." << std::endl;
    while (true) {
        // Read frame
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Empty frame received" << std::endl;
            break;
        }
        
        // Process frame
        ArmTracker::TrackingResult result;
        tracker.processFrame(frame, result);
        
        // Create debug visualization
        cv::Mat debug_frame = frame.clone();
        
        // Draw tracking status
        std::string status = result.trackingLost ? "Tracking Lost" : "Tracking Active";
        cv::putText(debug_frame, status, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.0,
                   result.trackingLost ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0),
                   2);
        
        // Draw joints
        for (const auto& [joint_name, joint_state] : result.joints) {
            // Draw joint position
            cv::circle(debug_frame, joint_state.pixelPos, 5, 
                      cv::Scalar(0, 255, 0), -1);
            
            // Draw joint name and confidence
            std::string info = joint_name + " (" + 
                             std::to_string(joint_state.confidence).substr(0, 4) + ")";
            cv::putText(debug_frame, info,
                       cv::Point(joint_state.pixelPos.x + 10, joint_state.pixelPos.y),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5,
                       cv::Scalar(255, 255, 255), 1);
        }
        
        // Show frame counter
        cv::putText(debug_frame, "Frame: " + std::to_string(frame_count++),
                   cv::Point(10, debug_frame.rows - 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5,
                   cv::Scalar(255, 255, 255), 1);
        
        // Show the frame
        cv::imshow("Tracking Debug", debug_frame);
        
        // Handle keyboard input
        int key = cv::waitKey(1);
        if (key == 27) break;  // ESC to exit
        else if (key == 'l' || key == 'L') tracker.toggleArm("left");
        else if (key == 'r' || key == 'R') tracker.toggleArm("right");
        else if (key == 'f' || key == 'F') {
            tracker.toggleFingers("left");
            tracker.toggleFingers("right");
        }
    }
    
    std::cout << "Cleaning up..." << std::endl;
    cap.release();
    cv::destroyAllWindows();
    return 0;
}