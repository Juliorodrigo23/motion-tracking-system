#include "clay_ui_wrapper.hpp"

void ClayUIWrapper::render(const cv::Mat& raw_frame, 
                          const cv::Mat& tracking_frame,
                          const ArmTracker::TrackingResult& result) {
    Clay_BeginLayout();

    CLAY(CLAY_ID("MainContainer"), 
         CLAY_LAYOUT({
             .sizing = { 
                 CLAY_SIZING_FIXED(static_cast<float>(window_width)), 
                 CLAY_SIZING_FIXED(static_cast<float>(window_height)) 
             },
             .padding = {16, 16},
             .childGap = 250,
             .layoutDirection = CLAY_TOP_TO_BOTTOM
         }),
         CLAY_RECTANGLE({.color = colors.background})) {
        
        CLAY(CLAY_ID("VideoPanelsContainer"),
             CLAY_LAYOUT({
                 .sizing = { 
                     CLAY_SIZING_GROW(1), 
                     CLAY_SIZING_FIXED(static_cast<float>(raw_frame.rows * 0.6))
                 },
                 .layoutDirection = CLAY_LEFT_TO_RIGHT,
                 .childGap = 16
             })) {
            
            renderVideoPanel("LeftPanel");
            renderVideoPanel("RightPanel");
        }

        CLAY(CLAY_ID("RotationContainer"),
             CLAY_LAYOUT({
                 .sizing = { CLAY_SIZING_GROW(1), CLAY_SIZING_GROW(1) },
                 .layoutDirection = CLAY_LEFT_TO_RIGHT,
                 .childGap = 16
             })) {
            renderRotationPanel(result);
        }
    }

    renderCommands = Clay_EndLayout();
}

void ClayUIWrapper::renderRotationPanel(const ArmTracker::TrackingResult& result) {
    CLAY(CLAY_ID("RotationPanel"), 
         CLAY_LAYOUT({
             .sizing = { CLAY_SIZING_GROW(1), CLAY_SIZING_GROW(1) },
             .padding = {0, 0},
             .layoutDirection = CLAY_LEFT_TO_RIGHT,
             .childGap = 16
         }),
         CLAY_RECTANGLE({ .color = colors.background })) {

        // Left side
        CLAY(CLAY_ID("LeftRotation"),
             CLAY_LAYOUT({
                 .sizing = { CLAY_SIZING_GROW(1), CLAY_SIZING_GROW(1) },
                 .layoutDirection = CLAY_TOP_TO_BOTTOM,
                 .childGap = 2
             })) {
            renderRotationInfo("Left", result.gestures.count("left") ? 
                             &result.gestures.at("left") : nullptr);
        }

        // Right side
        CLAY(CLAY_ID("RightRotation"),
             CLAY_LAYOUT({
                 .sizing = { CLAY_SIZING_GROW(1), CLAY_SIZING_GROW(1) },
                 .layoutDirection = CLAY_TOP_TO_BOTTOM,
                 .childGap = 2
             })) {
            renderRotationInfo("Right", result.gestures.count("right") ? 
                             &result.gestures.at("right") : nullptr);
        }
    }
}

void ClayUIWrapper::renderRotationInfo(const std::string& side, const ArmTracker::GestureState* gesture) {
    if (side == "Left") {
        CLAY(CLAY_ID("LeftRotationInfo"),
             CLAY_LAYOUT({
                 .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                 .padding = {16, 16},
                 .childGap = 2
             }),
             CLAY_RECTANGLE({ .color = colors.card })) {

            CLAY_TEXT(CLAY_STRING("Left Arm:"),
                     CLAY_TEXT_CONFIG({
                         .fontSize = 18,
                         .textColor = colors.title
                     }));

            if (gesture && gesture->type != "none") {
                // Use direct string literals based on conditions
                if (gesture->type == "supination") {
                    CLAY_TEXT(CLAY_STRING("Supination"),
                            CLAY_TEXT_CONFIG({
                                .fontSize = 24,
                                .textColor = colors.text
                            }));
                } else {
                    CLAY_TEXT(CLAY_STRING("Pronation"),
                            CLAY_TEXT_CONFIG({
                                .fontSize = 24,
                                .textColor = colors.text
                            }));
                }

                int confidence = static_cast<int>(gesture->confidence * 100);
                Clay_Color confColor = confidence > 70 ? colors.success :
                                     confidence > 40 ? colors.warning :
                                     colors.error;

                if (confidence > 70) {
                    CLAY_TEXT(CLAY_STRING("High Confidence"),
                            CLAY_TEXT_CONFIG({
                                .fontSize = 12,
                                .textColor = confColor
                            }));
                } else if (confidence > 40) {
                    CLAY_TEXT(CLAY_STRING("Medium Confidence"),
                            CLAY_TEXT_CONFIG({
                                .fontSize = 12,
                                .textColor = confColor
                            }));
                } else {
                    CLAY_TEXT(CLAY_STRING("Low Confidence"),
                            CLAY_TEXT_CONFIG({
                                .fontSize = 12,
                                .textColor = confColor
                            }));
                }

            } else {
                CLAY_TEXT(CLAY_STRING("No rotation detected"),
                        CLAY_TEXT_CONFIG({
                            .fontSize = 24,
                            .textColor = colors.text
                        }));
            }
        }
    } else {
        // Right side with similar structure
        CLAY(CLAY_ID("RightRotationInfo"),
             CLAY_LAYOUT({
                 .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                 .padding = {16, 16},
                 .childGap = 2
             }),
             CLAY_RECTANGLE({ .color = colors.card })) {

            CLAY_TEXT(CLAY_STRING("Right Arm:"),
                     CLAY_TEXT_CONFIG({
                         .fontSize = 18,
                         .textColor = colors.title
                     }));

            if (gesture && gesture->type != "none") {
                if (gesture->type == "supination") {
                    CLAY_TEXT(CLAY_STRING("Supination"),
                            CLAY_TEXT_CONFIG({
                                .fontSize = 24,
                                .textColor = colors.text
                            }));
                } else {
                    CLAY_TEXT(CLAY_STRING("Pronation"),
                            CLAY_TEXT_CONFIG({
                                .fontSize = 24,
                                .textColor = colors.text
                            }));
                }

                int confidence = static_cast<int>(gesture->confidence * 100);
                Clay_Color confColor = confidence > 70 ? colors.success :
                                     confidence > 40 ? colors.warning :
                                     colors.error;

                if (confidence > 70) {
                    CLAY_TEXT(CLAY_STRING("High Confidence"),
                            CLAY_TEXT_CONFIG({
                                .fontSize = 12,
                                .textColor = confColor
                            }));
                } else if (confidence > 40) {
                    CLAY_TEXT(CLAY_STRING("Medium Confidence"),
                            CLAY_TEXT_CONFIG({
                                .fontSize = 12,
                                .textColor = confColor
                            }));
                } else {
                    CLAY_TEXT(CLAY_STRING("Low Confidence"),
                            CLAY_TEXT_CONFIG({
                                .fontSize = 12,
                                .textColor = confColor
                            }));
                }

                
            } else {
                CLAY_TEXT(CLAY_STRING("No rotation detected"),
                        CLAY_TEXT_CONFIG({
                            .fontSize = 24,
                            .textColor = colors.text
                        }));
            }
        }
    }
}

void ClayUIWrapper::renderVideoPanel(const std::string& panelId) {
    const float VIDEO_CONTAINER_HEIGHT = window_height * 0.6f;  // Fixed height for video container

    if (panelId == "LeftPanel") {
        CLAY(CLAY_ID("LeftPanel"), 
             CLAY_LAYOUT({
                 .sizing = { CLAY_SIZING_GROW(1), CLAY_SIZING_FIXED(VIDEO_CONTAINER_HEIGHT) },
                 .layoutDirection = CLAY_TOP_TO_BOTTOM,
                 .childGap = 16,
                 .padding = {16, 16}  // Add padding to contain the video
             }),
             CLAY_RECTANGLE({ .color = colors.card })) {
            
            // Title section with fixed height
            CLAY(CLAY_ID("LeftPanelTitle"),
                 CLAY_LAYOUT({
                     .sizing = { CLAY_SIZING_GROW(1), CLAY_SIZING_FIXED(60) },
                     .padding = {16, 16},
                     .childAlignment = {
                         .x = CLAY_ALIGN_X_CENTER,
                         .y = CLAY_ALIGN_Y_CENTER
                     }
                 })) {
                CLAY_TEXT(CLAY_STRING("Raw Feed"),
                         CLAY_TEXT_CONFIG({
                             .fontSize = 28,
                             .fontId = FONT_ROBOTO,
                             .textColor = colors.title,
                             .letterSpacing = 2
                         }));
            }

            // Video container that takes remaining space
            CLAY(CLAY_ID("LeftPanelVideo"),
                 CLAY_LAYOUT({
                     .sizing = { CLAY_SIZING_GROW(1), CLAY_SIZING_GROW(1) },
                     .padding = {32, 16}
                 }),
                 CLAY_RECTANGLE({ .color = colors.background })) {}
        }
    } else {
        CLAY(CLAY_ID("RightPanel"), 
             CLAY_LAYOUT({
                 .sizing = { CLAY_SIZING_GROW(1), CLAY_SIZING_FIXED(VIDEO_CONTAINER_HEIGHT) },
                 .layoutDirection = CLAY_TOP_TO_BOTTOM,
                 .childGap = 16,
                 .padding = {16, 16}  // Add padding to contain the video
             }),
             CLAY_RECTANGLE({ .color = colors.card })) {
            
            CLAY(CLAY_ID("RightPanelTitle"),
                 CLAY_LAYOUT({
                     .sizing = { CLAY_SIZING_GROW(1), CLAY_SIZING_FIXED(60) },
                     .padding = {16, 16},
                     .childAlignment = {
                         .x = CLAY_ALIGN_X_CENTER,
                         .y = CLAY_ALIGN_Y_CENTER
                     }
                 })) {
                CLAY_TEXT(CLAY_STRING("Tracking Feed"),
                         CLAY_TEXT_CONFIG({
                             .fontSize = 28,
                             .fontId = FONT_ROBOTO,
                             .textColor = colors.title,
                             .letterSpacing = 2
                         }));
            }

            // Video container that takes remaining space
            CLAY(CLAY_ID("RightPanelVideo"),
                 CLAY_LAYOUT({
                     .sizing = { CLAY_SIZING_GROW(1), CLAY_SIZING_GROW(1) },
                     .padding = {16, 16}
                 }),
                 CLAY_RECTANGLE({ .color = colors.background })) {}
        }
    }
}