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
        
        Clay_Initialize(arena, (Clay_Dimensions){
            static_cast<float>(width), 
            static_cast<float>(height)
        }, {});
        
        Clay_SetMeasureTextFunction(MeasureText);
        
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

        // Main background container - grows with window
        CLAY(CLAY_ID("MainContainer"), 
             CLAY_LAYOUT({
                 .sizing = {
                     CLAY_SIZING_GROW(0),  // Grow to fill window width
                     CLAY_SIZING_GROW(0)   // Grow to fill window height
                 },
                 .padding = {16, 16},
                 .childGap = 16,
                 .layoutDirection = CLAY_LEFT_TO_RIGHT
             }),
             CLAY_RECTANGLE({.color = colors.background})) {
            
            // Left panel container
            CLAY(CLAY_ID("LeftPanel"), 
                 CLAY_LAYOUT({
                     .sizing = {
                         CLAY_SIZING_GROW(1),    // Take up half the space
                         CLAY_SIZING_GROW(0)     // Fill height
                     },
                     .layoutDirection = CLAY_TOP_TO_BOTTOM,
                     .childGap = 16
                 }),
                 CLAY_RECTANGLE({.color = colors.card})) {
                
                // Title container
                CLAY(CLAY_ID("LeftTitleContainer"),
                     CLAY_LAYOUT({
                         .sizing = {
                             CLAY_SIZING_GROW(0),  // Fill width of parent
                             CLAY_SIZING_FIT(0)    // Fit height to content
                         },
                         .padding = {16, 16},
                         .childAlignment = {
                             .x = CLAY_ALIGN_X_CENTER,
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

                // Video container
                CLAY(CLAY_ID("LeftVideoContainer"),
                     CLAY_LAYOUT({
                         .sizing = {
                             CLAY_SIZING_GROW(0),  // Fill width of parent
                             CLAY_SIZING_GROW(1)   // Take remaining height
                         },
                         .padding = {16, 16}
                     })) {
                    // Video feed will be rendered here
                }
            }
            
            // Right panel container
            CLAY(CLAY_ID("RightPanel"),
                 CLAY_LAYOUT({
                     .sizing = {
                         CLAY_SIZING_GROW(1),    // Take up half the space
                         CLAY_SIZING_GROW(0)     // Fill height
                     },
                     .layoutDirection = CLAY_TOP_TO_BOTTOM,
                     .childGap = 16
                 }),
                 CLAY_RECTANGLE({.color = colors.card})) {
                
                // Title container
                CLAY(CLAY_ID("RightTitleContainer"),
                     CLAY_LAYOUT({
                         .sizing = {
                             CLAY_SIZING_GROW(0),  // Fill width of parent
                             CLAY_SIZING_FIT(0)    // Fit height to content
                         },
                         .padding = {16, 16},
                         .childAlignment = {
                             .x = CLAY_ALIGN_X_CENTER,
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

                // Video container
                CLAY(CLAY_ID("RightVideoContainer"),
                     CLAY_LAYOUT({
                         .sizing = {
                             CLAY_SIZING_GROW(0),  // Fill width of parent
                             CLAY_SIZING_GROW(1)   // Take remaining height
                         },
                         .padding = {16, 16}
                     })) {
                    // Video feed will be rendered here
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
        float baseWidth = text->length * config->fontSize * 0.58f;
        float letterSpacingTotal = (text->length - 1) * config->letterSpacing;
        float totalWidth = baseWidth + letterSpacingTotal;
        float height = config->fontSize * 1.2f;
        
        return (Clay_Dimensions){
            static_cast<float>(totalWidth),
            static_cast<float>(height)
        };
    }
};