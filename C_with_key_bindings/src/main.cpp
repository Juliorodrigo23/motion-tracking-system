#include "arm_tracker.hpp"
#include "clay_ui_wrapper.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

// Helper function to calculate scaled dimensions while maintaining aspect ratio
cv::Size calculateScaledSize(const cv::Mat& frame, float scale) {
    int new_width = frame.cols * scale;
    int new_height = frame.rows * scale;
    return cv::Size(new_width, new_height);
}

// Font rendering functionality
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

// Global font data
stbtt_fontinfo font_info;
unsigned char* font_buffer = nullptr;

// Load font file and initialize stb_truetype
bool loadFont(const char* filename) {
    FILE* font_file = fopen(filename, "rb");
    if (!font_file) {
        std::cerr << "Failed to open font file: " << filename << std::endl;
        return false;
    }

    fseek(font_file, 0, SEEK_END);
    long size = ftell(font_file);
    fseek(font_file, 0, SEEK_SET);

    font_buffer = new unsigned char[size];
    fread(font_buffer, 1, size, font_file);
    fclose(font_file);

    if (stbtt_InitFont(&font_info, font_buffer, 0) == 0) {
        std::cerr << "Failed to initialize font" << std::endl;
        delete[] font_buffer;
        font_buffer = nullptr;
        return false;
    }

    return true;
}

// Clay command rendering
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

                // Create bitmap for text rendering
                float scale = stbtt_ScaleForPixelHeight(&font_info, config->fontSize);
                int ascent, descent, lineGap;
                stbtt_GetFontVMetrics(&font_info, &ascent, &descent, &lineGap);

                float scaled_ascent = ascent * scale;
                int x = rect.x;
                int y = rect.y + (rect.height + config->fontSize) / 2;

                // Render text
                std::string text_str(cmd->text.chars, cmd->text.length);
                for (char c : text_str) {
                    int w, h, xoff, yoff;
                    unsigned char* bitmap = stbtt_GetCodepointBitmap(&font_info, scale, scale,
                                                                c, &w, &h, &xoff, &yoff);

                    // Blend character bitmap onto output
                    for (int row = 0; row < h; row++) {
                        for (int col = 0; col < w; col++) {
                            unsigned char alpha = bitmap[row * w + col];
                            if (alpha > 0) {
                                int px = x + col + xoff;
                                int py = y + row + yoff - config->fontSize;
                                
                                if (px >= 0 && px < output.cols && py >= 0 && py < output.rows) {
                                    cv::Vec3b& pixel = output.at<cv::Vec3b>(py, px);
                                    pixel[0] = ((255 - alpha) * pixel[0] + alpha * config->textColor.b) / 255;
                                    pixel[1] = ((255 - alpha) * pixel[1] + alpha * config->textColor.g) / 255;
                                    pixel[2] = ((255 - alpha) * pixel[2] + alpha * config->textColor.r) / 255;
                                }
                            }
                        }
                    }

                    // Advance cursor
                    int advance, lsb;
                    stbtt_GetCodepointHMetrics(&font_info, c, &advance, &lsb);
                    x += advance * scale + config->letterSpacing;
                    stbtt_FreeBitmap(bitmap, nullptr);
                }
                break;
            }
            default:
                break;
        }
    }
}

int main() {
    std::cout << "Opening camera..." << std::endl;
    
    // Get executable path and construct font path
    std::filesystem::path exe_path = std::filesystem::current_path();
    std::string font_path = (exe_path / "../fonts/Roboto-Regular.ttf").string();
    
    std::cout << "Looking for font at: " << font_path << std::endl;
    
    if (!loadFont(font_path.c_str())) {
        std::cerr << "Failed to load font at: " << font_path << std::endl;
        return -1;
    }
    
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }
    
    // Get camera properties
    int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Camera resolution: " << frameWidth << "x" << frameHeight << std::endl;
    
    // Create initial frame to get dimensions
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Error: Could not get initial frame from camera" << std::endl;
        return -1;
    }
    
    // Calculate dimensions for scaled size while maintaining aspect ratio
    float scale = 0.25f;
    cv::Size scaledSize = calculateScaledSize(frame, scale);
    
    // Calculate UI dimensions with proper padding and space for all elements
    const int PADDING = 16;
    const int VIDEO_GAP = 40;
    const int TITLE_HEIGHT = 60;
    const int ROTATION_PANEL_HEIGHT = 200;
    
    // Calculate video panel dimensions to ensure containment
    const int VIDEO_PANEL_WIDTH = scaledSize.width + (PADDING * 1);  // Add padding on both sides
    const int VIDEO_PANEL_HEIGHT = TITLE_HEIGHT + scaledSize.height + (PADDING * 3.25);  // Add padding top and bottom
    
    // Calculate total UI dimensions
    const int UI_WIDTH = (VIDEO_PANEL_WIDTH * 2) + VIDEO_GAP + (PADDING * 2);  // Two panels plus gap
    const int UI_HEIGHT = VIDEO_PANEL_HEIGHT + ROTATION_PANEL_HEIGHT + (PADDING * 3);
    
    // Create UI wrapper with calculated dimensions
    ClayUIWrapper ui(UI_WIDTH, UI_HEIGHT);
    
    // Create tracker
    ArmTracker tracker;
    
    // Create output window
    cv::namedWindow("Arm Tracking", cv::WINDOW_NORMAL);
    cv::resizeWindow("Arm Tracking", UI_WIDTH, UI_HEIGHT);
    
    cv::Mat debug_frame, ui_frame;
    cv::Mat scaled_frame, scaled_debug;
    
    std::cout << "Scaled video size: " << scaledSize.width << "x" << scaledSize.height << std::endl;
    std::cout << "UI window size: " << UI_WIDTH << "x" << UI_HEIGHT << std::endl;
    
    while (true) {
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
        
        // Render Clay UI with tracking result
        ui.render(scaled_frame, scaled_debug, result);
        
        // Draw Clay commands
        renderClayCommands(ui_frame, ui.getRenderCommands());
        ui.overlayLogo(ui_frame);  
        //LEFT VIDEO
        // Position videos within their containers - adjust positioning to account for padding and title
        if (!scaled_frame.empty()) {
            int vertical_center = (UI_HEIGHT - scaledSize.height) / 2;
            scaled_frame.copyTo(ui_frame(cv::Rect(
                PADDING + PADDING,  // Container padding + internal padding
                vertical_center + PADDING*4,  // Container padding + title + internal padding
                scaledSize.width, 
                scaledSize.height
            )));
        }
        //RIGHT VIDEO   
        if (!scaled_debug.empty()) {
            int vertical_center = (UI_HEIGHT - scaledSize.height) / 2;
            scaled_debug.copyTo(ui_frame(cv::Rect(
                PADDING + VIDEO_PANEL_WIDTH + VIDEO_GAP,  // First panel + gap + paddings
                vertical_center + PADDING*4,
                scaledSize.width, 
                scaledSize.height
            )));
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

    // Cleanup
    if (font_buffer) {
        delete[] font_buffer;
    }
    cv::destroyAllWindows();
    return 0;
}

