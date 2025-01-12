#pragma once
#define CLAY_IMPLEMENTATION
#include "clay.h"
#include <opencv2/opencv.hpp>
#include <iostream>

enum FontIds {
    FONT_DEFAULT = 0,
    FONT_ROBOTO = 1
};

class ClayUIWrapper {
public:
    ClayUIWrapper(int width, int height) : window_width(width), window_height(height) {
        // Initialize Clay
        uint64_t memorySize = Clay_MinMemorySize();
        arena_memory = malloc(memorySize);
        arena = Clay_CreateArenaWithCapacityAndMemory(memorySize, arena_memory);
        
        // Initialize Clay with window dimensions
        Clay_Initialize(arena, (Clay_Dimensions){
            static_cast<float>(width), 
            static_cast<float>(height)
        }, {});
        
        // Set up text measurement
        Clay_SetMeasureTextFunction(MeasureText);
        
        // Initialize colors
        colors.background = (Clay_Color){32, 33, 36, 255};
        colors.card = (Clay_Color){15, 47, 62, 255};
        colors.accent = (Clay_Color){66, 133, 244, 255};
        colors.text = (Clay_Color){255, 255, 255, 255};
        colors.title = (Clay_Color){255, 255, 255, 255};
    }
    
    ~ClayUIWrapper() {
        if (arena_memory) {
            free(arena_memory);
        }
    }

    void render(const cv::Mat& raw_frame, const cv::Mat& tracking_frame, bool isTracking = false) {
        Clay_BeginLayout();

        // Main container
        CLAY(CLAY_ID("MainContainer"), 
             CLAY_LAYOUT({
                 .sizing = {
                     CLAY_SIZING_FIXED(static_cast<float>(window_width)), 
                     CLAY_SIZING_FIXED(static_cast<float>(window_height))
                 },
                 .padding = {16, 16},
                 .childGap = 16,
                 .layoutDirection = CLAY_LEFT_TO_RIGHT
             }),
             CLAY_RECTANGLE({.color = colors.background})) {
            
            // Left side (raw feed)
            CLAY(CLAY_ID("LeftPanel"), 
                 CLAY_LAYOUT({
                     .sizing = {
                         CLAY_SIZING_FIXED(static_cast<float>((window_width - 48) / 2)),
                         CLAY_SIZING_FIXED(static_cast<float>(window_height - 32))
                     },
                     .padding = {24, 24},
                     .childGap = 16,
                     .layoutDirection = CLAY_TOP_TO_BOTTOM
                 }),
                 CLAY_RECTANGLE({.color = colors.card})) {
                
                // Title section
                CLAY(CLAY_ID("LeftTitle"),
                     CLAY_LAYOUT({
                         .padding = {8, 8},
                         .childAlignment = {
                             .x = CLAY_ALIGN_X_LEFT,
                             .y = CLAY_ALIGN_Y_CENTER
                         }
                     })) {
                    CLAY_TEXT(CLAY_STRING("Raw Feed"),
                             CLAY_TEXT_CONFIG({
                                 .fontSize = static_cast<uint16_t>(28),
                                 .fontId = FONT_ROBOTO,
                                 .textColor = colors.title,
                                 .letterSpacing = 2
                             }));
                }
            }
            
            // Right side (tracking feed)
            CLAY(CLAY_ID("RightPanel"),
                 CLAY_LAYOUT({
                     .sizing = {
                         CLAY_SIZING_FIXED(static_cast<float>((window_width - 48) / 2)),
                         CLAY_SIZING_FIXED(static_cast<float>(window_height - 32))
                     },
                     .padding = {24, 24},
                     .childGap = 16,
                     .layoutDirection = CLAY_TOP_TO_BOTTOM
                 }),
                 CLAY_RECTANGLE({.color = colors.card})) {
                
                // Title section with tracking status
                CLAY(CLAY_ID("RightTitle"),
                     CLAY_LAYOUT({
                         .padding = {8, 8},
                         .childAlignment = {
                             .x = CLAY_ALIGN_X_LEFT,
                             .y = CLAY_ALIGN_Y_CENTER
                         }
                     })) {
                    CLAY_TEXT(CLAY_STRING("Tracking Feed"),
                             CLAY_TEXT_CONFIG({
                                 .fontSize = static_cast<uint16_t>(28),
                                 .fontId = FONT_ROBOTO,
                                 .textColor = colors.title,
                                 .letterSpacing = 2
                             }));
                }
                
                
            }
        }

        render_commands = Clay_EndLayout();
    }
    
    const Clay_RenderCommandArray& getRenderCommands() const {
        return render_commands;
    }

    int getWidth() const { return window_width; }
    int getHeight() const { return window_height; }

private:
    void* arena_memory;
    Clay_Arena arena;
    Clay_RenderCommandArray render_commands;
    int window_width;
    int window_height;

    struct {
        Clay_Color background;
        Clay_Color card;
        Clay_Color accent;
        Clay_Color text;
        Clay_Color title;
    } colors;

    static Clay_Dimensions MeasureText(Clay_String* text, Clay_TextElementConfig* config) {
        // Better text measurement with letter spacing consideration
        float baseWidth = text->length * config->fontSize * 0.58f;  // Roboto-specific scaling
        float letterSpacingTotal = (text->length - 1) * config->letterSpacing;
        float totalWidth = baseWidth + letterSpacingTotal;
        float height = config->fontSize * 1.2f;
        
        return (Clay_Dimensions){
            static_cast<float>(totalWidth),
            static_cast<float>(height)
        };
    }
};