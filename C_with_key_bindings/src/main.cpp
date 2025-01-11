#include "arm_tracker.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

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
    
    // Create windows for both views
    cv::namedWindow("Raw Feed");
    cv::namedWindow("Tracking Overlay");
    
    // Position windows side by side
    cv::moveWindow("Raw Feed", 50, 50);
    cv::moveWindow("Tracking Overlay", frameWidth + 100, 50);
    
    // Create tracker instance
    std::cout << "Initializing ArmTracker..." << std::endl;
    ArmTracker tracker;
    std::cout << "ArmTracker initialized" << std::endl;
    
    cv::Mat frame, debug_frame;
    int frame_count = 0;
    
    std::cout << "Starting main loop..." << std::endl;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Empty frame received" << std::endl;
            break;
        }
        
        // Show raw feed
        cv::imshow("Raw Feed", frame);
        
        // Process frame and get debug visualization
        ArmTracker::TrackingResult result;
        if (tracker.processFrame(frame, result, debug_frame)) {
            // Show overlay view
            cv::imshow("Tracking Overlay", debug_frame);
        }
        
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