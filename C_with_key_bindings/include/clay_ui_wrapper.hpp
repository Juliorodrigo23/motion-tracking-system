// clay_ui_wrapper.hpp
#pragma once
#include "clay.h"
#include "arm_tracker.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

enum FontIds {
    FONT_DEFAULT = 0,
    FONT_ROBOTO = 1
};

class ClayUIWrapper {
private:
    void* arena_memory;
    Clay_Arena arena;
    Clay_RenderCommandArray renderCommands;
    int window_width;
    int window_height;
    bool is_recording = false;
    cv::Mat logo;  // Store the logo as cv::Mat instead
    const int LOGO_WIDTH = 125;
    const int LOGO_HEIGHT = 100;
    
    struct {
        Clay_Color background;
        Clay_Color card;
        Clay_Color accent;
        Clay_Color text;
        Clay_Color title;
        Clay_Color success;
        Clay_Color warning;
        Clay_Color error;
    } colors;

public:
    ClayUIWrapper(int width, int height) : window_width(width), window_height(height) {
        uint64_t memorySize = Clay_MinMemorySize();
        arena_memory = malloc(memorySize);
        arena = Clay_CreateArenaWithCapacityAndMemory(memorySize, arena_memory);
        
        Clay_Initialize(arena, (Clay_Dimensions){
            static_cast<float>(width),
            static_cast<float>(height)
        }, {});
        
        Clay_SetMeasureTextFunction(MeasureText);
        
        // Initialize colors
        colors.background = (Clay_Color){0, 0, 0, 255};
        colors.card = (Clay_Color){15, 47, 62, 255};
        colors.accent = (Clay_Color){66, 133, 244, 255};
        colors.text = (Clay_Color){255, 255, 255, 255};
        colors.title = (Clay_Color){255, 255, 255, 255};
        colors.success = (Clay_Color){76, 175, 80, 255};
        colors.warning = (Clay_Color){255, 152, 0, 255};
        colors.error = (Clay_Color){244, 67, 54, 255};

        // Load logo
        loadLogo();
    }
    
    ~ClayUIWrapper() {
        if (arena_memory) {
            free(arena_memory);
        }
    }

    void toggleRecording() {
        is_recording = !is_recording;
    }
    
    void setRecordingState(bool recording) {
        is_recording = recording;
    }
    
    bool getRecordingState() const {
        return is_recording;
    }
    
    int getWidth() const {
        return window_width;
    }
    
    int getHeight() const {
        return window_height;
    }

    void render(const cv::Mat& raw_frame,
                const cv::Mat& tracking_frame,
                const ArmTracker::TrackingResult& result);
    
    const Clay_RenderCommandArray& getRenderCommands() const {
        return renderCommands;
    }

    // Method to overlay logo onto UI frame
    void overlayLogo(cv::Mat& ui_frame) {
        if (!logo.empty()) {
            // Logo position in UI frame (top-left corner with padding)
            int y_offset = (120 - LOGO_HEIGHT) / 2;  // Center in header height
            cv::Rect roi(32, y_offset, LOGO_WIDTH, LOGO_HEIGHT);
            
            // Check if ROI is within bounds
            if (roi.x >= 0 && roi.y >= 0 &&
                roi.x + roi.width <= ui_frame.cols &&
                roi.y + roi.height <= ui_frame.rows) {
                
                cv::Mat destinationROI = ui_frame(roi);

                if (logo.channels() == 4) {
                    // Process with alpha channel
                    for (int y = 0; y < logo.rows; y++) {
                        for (int x = 0; x < logo.cols; x++) {
                            cv::Vec4b& src = logo.at<cv::Vec4b>(y, x);
                            cv::Vec3b& dst = destinationROI.at<cv::Vec3b>(y, x);
                            
                            float alpha = src[3] / 255.0f;
                            dst[0] = (uchar)(src[0] * alpha + dst[0] * (1 - alpha));
                            dst[1] = (uchar)(src[1] * alpha + dst[1] * (1 - alpha));
                            dst[2] = (uchar)(src[2] * alpha + dst[2] * (1 - alpha));
                        }
                    }
                } else if (logo.channels() == 3) {
                    // Direct copy for RGB
                    logo.copyTo(destinationROI);
                } else {
                    std::cerr << "Unexpected number of channels in logo: " << logo.channels() << std::endl;
                }
            } else {
                std::cerr << "Logo ROI out of bounds" << std::endl;
            }
        }
    }

private:
    static Clay_Dimensions MeasureText(Clay_String* text, Clay_TextElementConfig* config) {
        float baseWidth = text->length * config->fontSize * 0.58f;
        float letterSpacingTotal = (text->length - 1) * config->letterSpacing;
        float totalWidth = baseWidth + letterSpacingTotal;
        float height = config->lineHeight > 0 ? config->lineHeight : (config->fontSize * 1.2f);
        
        return (Clay_Dimensions){
            totalWidth,
            height
        };
    }

    void loadLogo() {
        const std::vector<std::string> possible_paths = {
            "../include/supro.png",
            "../../include/supro.png",
            "include/supro.png",
            "supro.png",
            "./supro.png"
        };
        
        std::string successful_path;
        
        for (const auto& path : possible_paths) {
            std::cout << "Trying logo path: " << path << std::endl;
            this->logo = cv::imread(path, cv::IMREAD_UNCHANGED);
            if (!this->logo.empty()) {
                successful_path = path;
                std::cout << "Successfully loaded logo from: " << path << std::endl;
                std::cout << "Logo dimensions: " << logo.size() << " channels: " << logo.channels() << std::endl;
                break;
            }
        }
        
        if (this->logo.empty()) {
            std::cerr << "Warning: Failed to load logo PNG from any attempted path" << std::endl;
            return;
        }

        // Resize to our desired dimensions
        cv::resize(this->logo, this->logo, cv::Size(LOGO_WIDTH, LOGO_HEIGHT), 0, 0, cv::INTER_LINEAR);
    }

    void renderVideoPanel(const std::string& panelId);
    void renderRotationPanel(const ArmTracker::TrackingResult& result);
    void renderRotationInfo(const std::string& side, const ArmTracker::GestureState* gesture);
};