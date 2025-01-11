#include "arm_tracker.hpp"
#include "clay_ui_wrapper.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

// Helper function to calculate scaled dimensions while maintaining aspect ratio
cv::Size calculateScaledSize(const cv::Mat& frame, float scale) {
    int new_width = frame.cols * scale;
    int new_height = frame.rows * scale;
    return cv::Size(new_width, new_height);
}

void renderClayCommands(cv::Mat& output, const Clay_RenderCommandArray& commands) {
    for (int i = 0; i < commands.length; i++) {
        const Clay_RenderCommand* cmd = &commands.internalArray[i];
        
        switch (cmd->commandType) {
            case CLAY_RENDER_COMMAND_TYPE_RECTANGLE: {
                const auto& rect = cmd->boundingBox;
                const auto& color = cmd->config.rectangleElementConfig->color;
                cv::rectangle(output, 
                            cv::Rect(rect.x, rect.y, rect.width, rect.height),
                            cv::Scalar(color.b, color.g, color.r, color.a), 
                            cv::FILLED);
                break;
            }
            case CLAY_RENDER_COMMAND_TYPE_TEXT: {
                const auto& rect = cmd->boundingBox;
                const auto& config = cmd->config.textElementConfig;
                cv::putText(output,
                           std::string(cmd->text.chars, cmd->text.length),
                           cv::Point(rect.x, rect.y + rect.height),
                           cv::FONT_HERSHEY_SIMPLEX,
                           config->fontSize / 24.0f,
                           cv::Scalar(config->textColor.b, config->textColor.g, 
                                    config->textColor.r, config->textColor.a),
                           1, cv::LINE_AA);
                break;
            }
            default:
                break;
        }
    }
}

int main() {
    std::cout << "Opening camera..." << std::endl;
    
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }
    
    // Get camera properties
    int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Camera resolution: " << frameWidth << "x" << frameHeight << std::endl;
    
    // Calculate dimensions for 1/4 size while maintaining aspect ratio
    float scale = 0.25f;  // 1/4 size
    cv::Size scaledSize = cv::Size(frameWidth * scale, frameHeight * scale);
    
    // Create UI with fixed size based on scaled video dimensions plus padding
    const int UI_WIDTH = (scaledSize.width * 2) + 80;  // Space for two videos side by side plus padding
    const int UI_HEIGHT = scaledSize.height + 96;      // Video height plus padding
    ClayUIWrapper ui(UI_WIDTH, UI_HEIGHT);
    
    // Create tracker
    ArmTracker tracker;
    
    // Create output window
    cv::namedWindow("Arm Tracking", cv::WINDOW_NORMAL);
    cv::resizeWindow("Arm Tracking", UI_WIDTH, UI_HEIGHT);
    
    cv::Mat frame, debug_frame, ui_frame;
    cv::Mat scaled_frame, scaled_debug;
    
    std::cout << "Scaled video size: " << scaledSize.width << "x" << scaledSize.height << std::endl;
    std::cout << "UI window size: " << UI_WIDTH << "x" << UI_HEIGHT << std::endl;
    
    while (true) {
        // Capture frame
        cap >> frame;
        if (frame.empty()) break;
        
        // Create UI background
        ui_frame = cv::Mat(UI_HEIGHT, UI_WIDTH, CV_8UC3, cv::Scalar(32, 33, 36));
        
        // Process frame for tracking
        ArmTracker::TrackingResult result;
        tracker.processFrame(frame, result, debug_frame);
        
        // Scale frames maintaining aspect ratio
        cv::resize(frame, scaled_frame, scaledSize, 0, 0, cv::INTER_AREA);
        if (!debug_frame.empty()) {
            cv::resize(debug_frame, scaled_debug, scaledSize, 0, 0, cv::INTER_AREA);
        }
        
        // Render Clay UI
        ui.render(scaled_frame, scaled_debug);
        
        // Draw Clay commands
        renderClayCommands(ui_frame, ui.getRenderCommands());
        
        // Calculate positions for video frames
        int y_offset = 48;  // Top padding
        int x_gap = 40;     // Gap between videos
        
        // Copy camera frames into UI frame
        if (!scaled_frame.empty()) {
            scaled_frame.copyTo(ui_frame(cv::Rect(24, y_offset, 
                                                scaledSize.width, 
                                                scaledSize.height)));
        }
        if (!scaled_debug.empty()) {
            scaled_debug.copyTo(ui_frame(cv::Rect(24 + scaledSize.width + x_gap, y_offset, 
                                                scaledSize.width, 
                                                scaledSize.height)));
        }
        
        // Show combined frame
        cv::imshow("Arm Tracking", ui_frame);
        
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
    
    cv::destroyAllWindows();
    return 0;
}