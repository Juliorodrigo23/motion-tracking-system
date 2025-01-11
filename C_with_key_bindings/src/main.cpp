#include "arm_tracker.hpp"
#include "ui_wrapper.hpp"
#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }
    
    int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    // Create instances
    ArmTracker tracker;
    UIWrapper ui(frameWidth, frameHeight);
    
    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        // Get mouse state
        cv::Point mouse = cv::getWindowPoint("Tracking");
        int mouseFlags = cv::getWindowProperty("Tracking", cv::WND_PROP_MOUSE_CALLBACK);
        bool mouseDown = (mouseFlags & cv::EVENT_FLAG_LBUTTON) != 0;
        
        // Process frame with arm tracker
        ArmTracker::TrackingResult result;
        tracker.processFrame(frame, result);
        
        // Update UI
        ui.update(result, frame, 
                 static_cast<float>(mouse.x), 
                 static_cast<float>(mouse.y), 
                 mouseDown);
        
        // Render UI
        const auto& commands = ui.getRenderCommands();
        cv::Mat display = frame.clone();
        
        for (int i = 0; i < commands.length; i++) {
            const Clay_RenderCommand* cmd = &commands.internalArray[i];
            
            switch (cmd->commandType) {
                case CLAY_RENDER_COMMAND_TYPE_RECTANGLE: {
                    const auto& rect = cmd->boundingBox;
                    const auto& color = cmd->config.rectangleElementConfig->color;
                    cv::rectangle(display,
                                cv::Rect(rect.x, rect.y, rect.width, rect.height),
                                cv::Scalar(color.b, color.g, color.r, color.a),
                                cv::FILLED);
                    break;
                }
                case CLAY_RENDER_COMMAND_TYPE_TEXT: {
                    const auto& text = cmd->config.textElementConfig->text;
                    const auto& color = cmd->config.textElementConfig->textColor;
                    cv::putText(display,
                              std::string(text.chars, text.length),
                              cv::Point(cmd->boundingBox.x, 
                                      cmd->boundingBox.y + cmd->boundingBox.height),
                              cv::FONT_HERSHEY_SIMPLEX,
                              cmd->config.textElementConfig->fontSize / 24.0,
                              cv::Scalar(color.b, color.g, color.r, color.a));
                    break;
                }
            }
        }
        
        cv::imshow("Tracking", display);
        
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