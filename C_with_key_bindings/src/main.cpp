#include "arm_tracker.hpp"
#include "visualizer.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }
    
    int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    ArmTracker tracker;
    if (!tracker.initialize()) {
        std::cerr << "Error: Could not initialize MediaPipe" << std::endl;
        return -1;
    }

    TrackerVisualizer visualizer(frameWidth, frameHeight);
    
    cv::Mat frame, displayFrame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        frame.copyTo(displayFrame);
        
        ArmTracker::TrackingResult result;
        tracker.processFrame(frame, result);
        
        if (!result.trackingLost) {
            visualizer.drawOverlay(displayFrame, result);
        }
        
        // Show both original and tracked views
        cv::imshow("Original", frame);
        cv::imshow("Tracking", displayFrame);
        
        int key = cv::waitKey(1);
        if (key == 27) break;  // ESC to exit
        else if (key == 'l' || key == 'L') tracker.toggleArm("left");
        else if (key == 'r' || key == 'R') tracker.toggleArm("right");
        else if (key == 'f' || key == 'F') {
            tracker.toggleFingers("left");
            tracker.toggleFingers("right");
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    return 0;
}