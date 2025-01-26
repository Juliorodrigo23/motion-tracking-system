// ui_wrapper.hpp
#pragma once

#define CLAY_IMPLEMENTATION
#include "clay.h"
#include "arm_tracker.hpp"
#include <functional>

class UIWrapper {
public:
    UIWrapper(int width, int height) {
        // Initialize Clay
        uint64_t memorySize = Clay_MinMemorySize();
        arenaMemory = malloc(memorySize);
        arena = Clay_CreateArenaWithCapacityAndMemory(memorySize, arenaMemory);
        Clay_Initialize(arena, (Clay_Dimensions){(uint32_t)width, (uint32_t)height});
        
        Clay_SetMeasureTextFunction(MeasureText);
        
        // Initialize colors
        colors.background = (Clay_Color){240, 240, 245, 255};
        colors.sidebar = (Clay_Color){220, 220, 225, 255};
        colors.accent = (Clay_Color){66, 135, 245, 255};
        colors.text = (Clay_Color){50, 50, 55, 255};
    }
    
    ~UIWrapper() {
        if (arenaMemory) {
            free(arenaMemory);
        }
    }

    void update(const ArmTracker::TrackingResult& result, 
                const cv::Mat& frame,
                float mouseX, float mouseY, 
                bool mouseDown) {
        // Update Clay's pointer state
        Clay_SetPointerState((Clay_Vector2){mouseX, mouseY}, mouseDown);
        
        // Begin layout
        Clay_BeginLayout();
        
        // Main container
        CLAY(CLAY_ID("MainContainer"), 
            CLAY_LAYOUT({
                .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0)},
                .layoutDirection = CLAY_LEFT_TO_RIGHT
            }), 
            CLAY_RECTANGLE({.color = colors.background})) {
            
            // Sidebar
            CLAY(CLAY_ID("Sidebar"),
                CLAY_LAYOUT({
                    .sizing = {CLAY_SIZING_FIXED(300), CLAY_SIZING_GROW(0)},
                    .padding = {16, 16},
                    .childGap = 16,
                    .layoutDirection = CLAY_TOP_TO_BOTTOM
                }),
                CLAY_RECTANGLE({.color = colors.sidebar})) {
                
                // Title
                CLAY_TEXT(CLAY_STRING("Arm Tracking"), 
                        CLAY_TEXT_CONFIG({
                            .fontSize = 24,
                            .textColor = colors.text
                        }));
                
                // Status
                CLAY(CLAY_LAYOUT({.padding = {8, 8}}),
                    CLAY_RECTANGLE({
                        .color = result.trackingLost ? 
                                (Clay_Color){255, 100, 100, 255} : 
                                (Clay_Color){100, 255, 100, 255}
                    })) {
                    CLAY_TEXT(
                        CLAY_STRING(result.trackingLost ? "Tracking Lost" : "Tracking Active"),
                        CLAY_TEXT_CONFIG({
                            .fontSize = 16,
                            .textColor = {255, 255, 255, 255}
                        })
                    );
                }
                
                // Joint information
                for (const auto& [joint_name, joint_state] : result.joints) {
                    CLAY(CLAY_LAYOUT({
                            .padding = {8, 8},
                            .childGap = 4,
                            .layoutDirection = CLAY_TOP_TO_BOTTOM
                        }),
                        CLAY_RECTANGLE({.color = colors.accent})) {
                        // Joint name
                        CLAY_TEXT(
                            CLAY_STRING(joint_name.c_str()),
                            CLAY_TEXT_CONFIG({
                                .fontSize = 16,
                                .textColor = {255, 255, 255, 255}
                            })
                        );
                        
                        // Confidence score
                        char conf_text[32];
                        snprintf(conf_text, sizeof(conf_text), 
                                "Confidence: %.2f", joint_state.confidence);
                        CLAY_TEXT(
                            CLAY_STRING(conf_text),
                            CLAY_TEXT_CONFIG({
                                .fontSize = 14,
                                .textColor = {255, 255, 255, 255}
                            })
                        );
                    }
                }
            }
            
            // Main content area
            CLAY(CLAY_ID("Content"),
                CLAY_LAYOUT({
                    .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0)},
                    .padding = {16, 16}
                })) {
                // Here you would render your OpenCV frame
                // You'll need to implement custom rendering for the video frame
            }
        }
        
        // End layout and get render commands
        renderCommands = Clay_EndLayout();
    }
    
    const Clay_RenderCommandArray& getRenderCommands() const {
        return renderCommands;
    }

private:
    void* arenaMemory;
    Clay_Arena arena;
    Clay_RenderCommandArray renderCommands;
    
    struct {
        Clay_Color background;
        Clay_Color sidebar;
        Clay_Color accent;
        Clay_Color text;
    } colors;
    
    // Text measurement function
    static Clay_Dimensions MeasureText(Clay_String* text, Clay_TextElementConfig* config) {
        // You'll need to implement this based on your font rendering system
        // For now, return a simple approximation
        return (Clay_Dimensions){
            (uint32_t)(text->length * config->fontSize * 0.6f),
            (uint32_t)(config->fontSize * 1.2f)
        };
    }
};