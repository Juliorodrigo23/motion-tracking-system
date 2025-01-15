#pragma once
#include "clay.h"
#include "arm_tracker.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

enum FontIds {
    FONT_DEFAULT = 0,
    FONT_ROBOTO = 1
};

class ClayUIWrapper {
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
        
        colors.background = (Clay_Color){0, 0, 0, 255};
        colors.card = (Clay_Color){15, 47, 62, 255};
        colors.accent = (Clay_Color){66, 133, 244, 255};
        colors.text = (Clay_Color){255, 255, 255, 255};
        colors.title = (Clay_Color){255, 255, 255, 255};
        colors.success = (Clay_Color){76, 175, 80, 255};
        colors.warning = (Clay_Color){255, 152, 0, 255};
        colors.error = (Clay_Color){244, 67, 54, 255};
    }
    
    ~ClayUIWrapper() {
        if (arena_memory) {
            free(arena_memory);
        }
    }

    void render(const cv::Mat& raw_frame, 
                const cv::Mat& tracking_frame,
                const ArmTracker::TrackingResult& result);
    
    const Clay_RenderCommandArray& getRenderCommands() const {
        return renderCommands;
    }

private:
    void renderRotationPanel(const ArmTracker::TrackingResult& result);
    void renderRotationInfo(const std::string& side, const ArmTracker::GestureState* gesture);
    void renderVideoPanel(const std::string& panelId);

    static Clay_Dimensions MeasureText(Clay_String* text, Clay_TextElementConfig* config) {
        float baseWidth = text->length * config->fontSize * 0.58f;
        float letterSpacingTotal = (text->length - 1) * config->letterSpacing;
        float totalWidth = baseWidth + letterSpacingTotal;
        float height = config->fontSize * 1.2f;
        
        return (Clay_Dimensions){
            totalWidth,
            height
        };
    }

    void* arena_memory;
    Clay_Arena arena;
    Clay_RenderCommandArray renderCommands;
    int window_width;
    int window_height;

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
};