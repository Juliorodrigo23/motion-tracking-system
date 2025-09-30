// main.cpp
#include "arm_tracker.hpp"
#include "clay_ui_wrapper.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <fstream>

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

// Generate timestamp string for file naming
std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    return ss.str();
}

// Create output directory
std::filesystem::path createOutputDirectory(const std::string& base_name = "") {
    std::string dir_name = "output_" + (base_name.empty() ? "" : base_name + "_") + getTimestamp();
    std::filesystem::path output_dir = std::filesystem::current_path() / "outputs" / dir_name;
    std::filesystem::create_directories(output_dir);
    return output_dir;
}

// Save tracking data to CSV
void saveTrackingData(const std::filesystem::path& output_dir, 
                     const std::vector<ArmTracker::TrackingResult>& tracking_data,
                     const std::vector<double>& timestamps) {
    std::ofstream csv(output_dir / "tracking_data.csv");
    
    // Write header
    csv << "timestamp,frame,tracking_lost";
    csv << ",left_shoulder_x,left_shoulder_y,left_shoulder_z,left_shoulder_confidence";
    csv << ",right_shoulder_x,right_shoulder_y,right_shoulder_z,right_shoulder_confidence";
    csv << ",left_elbow_x,left_elbow_y,left_elbow_z,left_elbow_confidence";
    csv << ",right_elbow_x,right_elbow_y,right_elbow_z,right_elbow_confidence";
    csv << ",left_wrist_x,left_wrist_y,left_wrist_z,left_wrist_confidence";
    csv << ",right_wrist_x,right_wrist_y,right_wrist_z,right_wrist_confidence";
    csv << ",left_gesture,left_gesture_confidence,left_gesture_angle";
    csv << ",right_gesture,right_gesture_confidence,right_gesture_angle\n";
    
    // Write data
    for (size_t i = 0; i < tracking_data.size(); ++i) {
        const auto& result = tracking_data[i];
        csv << timestamps[i] << "," << i << "," << (result.trackingLost ? "1" : "0");
        
        // Joint data
        const std::vector<std::string> joint_names = {
            "left_shoulder", "right_shoulder", "left_elbow", 
            "right_elbow", "left_wrist", "right_wrist"
        };
        
        for (const auto& joint_name : joint_names) {
            if (result.joints.count(joint_name)) {
                const auto& joint = result.joints.at(joint_name);
                csv << "," << joint.position[0] << "," << joint.position[1] 
                    << "," << joint.position[2] << "," << joint.confidence;
            } else {
                csv << ",,,,";
            }
        }
        
        // Gesture data
        for (const std::string& side : {"left", "right"}) {
            if (result.gestures.count(side)) {
                const auto& gesture = result.gestures.at(side);
                csv << "," << gesture.type << "," << gesture.confidence 
                    << "," << gesture.angle;
            } else {
                csv << ",none,0,0";
            }
        }
        csv << "\n";
    }
    
    csv.close();
    std::cout << "Saved tracking data to: " << (output_dir / "tracking_data.csv") << std::endl;
}

// Process video file
void processVideoFile(const std::string& video_path, ArmTracker& tracker, ClayUIWrapper& ui) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file: " << video_path << std::endl;
        return;
    }
    
    // Get video properties
    int fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    std::cout << "Processing video: " << video_path << std::endl;
    std::cout << "Resolution: " << frame_width << "x" << frame_height << std::endl;
    std::cout << "FPS: " << fps << ", Total frames: " << total_frames << std::endl;
    
    // Create output directory
    std::filesystem::path video_name = std::filesystem::path(video_path).stem();
    std::filesystem::path output_dir = createOutputDirectory(video_name.string());
    
    // Calculate dimensions
    float scale = 0.25f;
    cv::Size scaledSize = calculateScaledSize(cv::Mat(frame_height, frame_width, CV_8UC3), scale);
    
    // Setup video writers
    cv::VideoWriter raw_writer(
        (output_dir / "original.mp4").string(),
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps, cv::Size(frame_width, frame_height)
    );
    
    cv::VideoWriter overlay_writer(
        (output_dir / "with_overlay.mp4").string(),
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps, cv::Size(ui.getWidth(), ui.getHeight())
    );
    
    // Process frames
    cv::Mat frame, debug_frame, ui_frame, scaled_frame, scaled_debug;
    std::vector<ArmTracker::TrackingResult> tracking_data;
    std::vector<double> timestamps;
    
    int frame_count = 0;
    while (cap.read(frame)) {
        // Track time
        double timestamp = frame_count / (double)fps;
        timestamps.push_back(timestamp);
        
        // Process frame
        ArmTracker::TrackingResult result;
        tracker.processFrame(frame, result, debug_frame);
        tracking_data.push_back(result);
        
        // Save original frame
        raw_writer.write(frame);
        
        // Create UI visualization
        ui_frame = cv::Mat(ui.getHeight(), ui.getWidth(), CV_8UC3, cv::Scalar(32, 33, 36));
        
        // Scale frames
        cv::resize(frame, scaled_frame, scaledSize, 0, 0, cv::INTER_AREA);
        if (!debug_frame.empty()) {
            cv::resize(debug_frame, scaled_debug, scaledSize, 0, 0, cv::INTER_AREA);
        }
        
        // Render UI
        ui.render(scaled_frame, scaled_debug, result);
        renderClayCommands(ui_frame, ui.getRenderCommands());
        ui.overlayLogo(ui_frame);
        
        // Overlay videos on UI
        if (!scaled_frame.empty()) {
            int vertical_center = (ui.getHeight() - scaledSize.height) / 2;
            scaled_frame.copyTo(ui_frame(cv::Rect(
                32, vertical_center + 64, scaledSize.width, scaledSize.height
            )));
        }
        
        if (!scaled_debug.empty()) {
            int vertical_center = (ui.getHeight() - scaledSize.height) / 2;
            scaled_debug.copyTo(ui_frame(cv::Rect(
                48 + scaledSize.width + 40, vertical_center + 64,
                scaledSize.width, scaledSize.height
            )));
        }
        
        // Save UI frame
        overlay_writer.write(ui_frame);
        
        // Progress
        frame_count++;
        if (frame_count % 30 == 0) {
            std::cout << "Progress: " << frame_count << "/" << total_frames 
                     << " (" << (frame_count * 100 / total_frames) << "%)" << std::endl;
        }
        
        // Show preview (optional - press ESC to skip preview)
        cv::imshow("Processing Video", ui_frame);
        if (cv::waitKey(1) == 27) break;
    }
    
    // Release writers
    raw_writer.release();
    overlay_writer.release();
    cap.release();
    
    // Save tracking data
    saveTrackingData(output_dir, tracking_data, timestamps);
    
    std::cout << "Processing complete! Output saved to: " << output_dir << std::endl;
}

int main(int argc, char* argv[]) {
    // Check for video file argument
    bool process_video_mode = false;
    std::string video_path;
    
    if (argc > 1) {
        video_path = argv[1];
        if (std::filesystem::exists(video_path)) {
            process_video_mode = true;
            std::cout << "Video processing mode: " << video_path << std::endl;
        } else {
            std::cerr << "Video file not found: " << video_path << std::endl;
            return -1;
        }
    }
    
    std::cout << "Initializing..." << std::endl;
    
    // Get executable path and construct font path
    std::filesystem::path exe_path = std::filesystem::current_path();
    std::string font_path = (exe_path / "../fonts/Roboto-Regular.ttf").string();
    
    std::cout << "Looking for font at: " << font_path << std::endl;
    
    if (!loadFont(font_path.c_str())) {
        std::cerr << "Failed to load font at: " << font_path << std::endl;
        return -1;
    }
    
    // Create tracker
    ArmTracker tracker;
    
    if (process_video_mode) {
        // Process video file
        cv::Mat sample_frame = cv::imread(video_path);
        if (sample_frame.empty()) {
            cv::VideoCapture temp_cap(video_path);
            temp_cap >> sample_frame;
            temp_cap.release();
        }
        
        float scale = 0.25f;
        cv::Size scaledSize = calculateScaledSize(sample_frame, scale);
        
        const int PADDING = 16;
        const int VIDEO_GAP = 40;
        const int TITLE_HEIGHT = 60;
        const int ROTATION_PANEL_HEIGHT = 200;
        
        const int VIDEO_PANEL_WIDTH = scaledSize.width + (PADDING * 1);
        const int VIDEO_PANEL_HEIGHT = TITLE_HEIGHT + scaledSize.height + (PADDING * 3.25);
        
        const int UI_WIDTH = (VIDEO_PANEL_WIDTH * 2) + VIDEO_GAP + (PADDING * 2);
        const int UI_HEIGHT = VIDEO_PANEL_HEIGHT + ROTATION_PANEL_HEIGHT + (PADDING * 3);
        
        ClayUIWrapper ui(UI_WIDTH, UI_HEIGHT);
        processVideoFile(video_path, tracker, ui);
    } else {
        // Live camera mode with recording capability
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
        
        // Create initial frame to get dimensions
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Could not get initial frame from camera" << std::endl;
            return -1;
        }
        
        // Calculate dimensions
        float scale = 0.25f;
        cv::Size scaledSize = calculateScaledSize(frame, scale);
        
        const int PADDING = 16;
        const int VIDEO_GAP = 40;
        const int TITLE_HEIGHT = 60;
        const int ROTATION_PANEL_HEIGHT = 200;
        
        const int VIDEO_PANEL_WIDTH = scaledSize.width + (PADDING * 1);
        const int VIDEO_PANEL_HEIGHT = TITLE_HEIGHT + scaledSize.height + (PADDING * 3.25);
        
        const int UI_WIDTH = (VIDEO_PANEL_WIDTH * 2) + VIDEO_GAP + (PADDING * 2);
        const int UI_HEIGHT = VIDEO_PANEL_HEIGHT + ROTATION_PANEL_HEIGHT + (PADDING * 3);
        
        // Create UI wrapper
        ClayUIWrapper ui(UI_WIDTH, UI_HEIGHT);
        
        // Create output window
        cv::namedWindow("Arm Tracking", cv::WINDOW_NORMAL);
        cv::resizeWindow("Arm Tracking", UI_WIDTH, UI_HEIGHT);
        
        cv::Mat debug_frame, ui_frame;
        cv::Mat scaled_frame, scaled_debug;
        
        // Recording state
        bool is_recording = false;
        cv::VideoWriter raw_writer, overlay_writer;
        std::vector<ArmTracker::TrackingResult> recording_data;
        std::vector<double> recording_timestamps;
        std::filesystem::path current_output_dir;
        auto recording_start_time = std::chrono::steady_clock::now();
        
        std::cout << "Controls:" << std::endl;
        std::cout << "  ESC - Exit" << std::endl;
        std::cout << "  Space - Start/Stop recording" << std::endl;
        std::cout << "  L - Toggle left arm" << std::endl;
        std::cout << "  R - Toggle right arm" << std::endl;
        std::cout << "  F - Toggle fingers" << std::endl;
        
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            
            // Create UI background
            ui_frame = cv::Mat(UI_HEIGHT, UI_WIDTH, CV_8UC3, cv::Scalar(32, 33, 36));
            
            // Process frame for tracking
            ArmTracker::TrackingResult result;
            tracker.processFrame(frame, result, debug_frame);
            
            // Handle recording
            if (is_recording) {
                raw_writer.write(frame);
                recording_data.push_back(result);
                
                auto current_time = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed = current_time - recording_start_time;
                recording_timestamps.push_back(elapsed.count());
            }
            
            // Scale frames
            cv::resize(frame, scaled_frame, scaledSize, 0, 0, cv::INTER_AREA);
            if (!debug_frame.empty()) {
                cv::resize(debug_frame, scaled_debug, scaledSize, 0, 0, cv::INTER_AREA);
            }
            
            // Render Clay UI
            ui.setRecordingState(is_recording);
            ui.render(scaled_frame, scaled_debug, result);
            
            // Draw Clay commands
            renderClayCommands(ui_frame, ui.getRenderCommands());
            ui.overlayLogo(ui_frame);
            
            // Position videos
            if (!scaled_frame.empty()) {
                int vertical_center = (UI_HEIGHT - scaledSize.height) / 2;
                scaled_frame.copyTo(ui_frame(cv::Rect(
                    PADDING + PADDING,
                    vertical_center + PADDING*4,
                    scaledSize.width,
                    scaledSize.height
                )));
            }
            
            if (!scaled_debug.empty()) {
                int vertical_center = (UI_HEIGHT - scaledSize.height) / 2;
                scaled_debug.copyTo(ui_frame(cv::Rect(
                    PADDING + VIDEO_PANEL_WIDTH + VIDEO_GAP,
                    vertical_center + PADDING*4,
                    scaledSize.width,
                    scaledSize.height
                )));
            }
            
            // Save overlay frame if recording
            if (is_recording && overlay_writer.isOpened()) {
                overlay_writer.write(ui_frame);
            }
            
            // Show combined frame
            cv::imshow("Arm Tracking", ui_frame);
            
            // Handle keyboard input
            int key = cv::waitKey(1);
            if (key == 27) break;  // ESC to exit
            else if (key == ' ') {  // Space to toggle recording
                if (!is_recording) {
                    // Start recording
                    is_recording = true;
                    current_output_dir = createOutputDirectory("live_recording");
                    
                    raw_writer.open(
                        (current_output_dir / "original.mp4").string(),
                        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        30, cv::Size(frameWidth, frameHeight)
                    );
                    
                    overlay_writer.open(
                        (current_output_dir / "with_overlay.mp4").string(),
                        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        30, cv::Size(UI_WIDTH, UI_HEIGHT)
                    );
                    
                    recording_data.clear();
                    recording_timestamps.clear();
                    recording_start_time = std::chrono::steady_clock::now();
                    
                    std::cout << "Recording started! Output will be saved to: " << current_output_dir << std::endl;
                } else {
                    // Stop recording
                    is_recording = false;
                    
                    if (raw_writer.isOpened()) raw_writer.release();
                    if (overlay_writer.isOpened()) overlay_writer.release();
                    
                    // Save tracking data
                    saveTrackingData(current_output_dir, recording_data, recording_timestamps);
                    
                    std::cout << "Recording stopped! Files saved to: " << current_output_dir << std::endl;
                }
            }
            else if (key == 'l' || key == 'L') tracker.toggleArm("left");
            else if (key == 'r' || key == 'R') tracker.toggleArm("right");
            else if (key == 'f' || key == 'F') {
                tracker.toggleFingers("left");
                tracker.toggleFingers("right");
            }
        }
        
        // Clean up any active recording
        if (is_recording) {
            if (raw_writer.isOpened()) raw_writer.release();
            if (overlay_writer.isOpened()) overlay_writer.release();
            saveTrackingData(current_output_dir, recording_data, recording_timestamps);
            std::cout << "Recording saved to: " << current_output_dir << std::endl;
        }
        
        cv::destroyAllWindows();
    }
    
    // Cleanup
    if (font_buffer) {
        delete[] font_buffer;
    }
    
    return 0;
}