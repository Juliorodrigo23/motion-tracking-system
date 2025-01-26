#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <iostream>

namespace py = pybind11;

class MediaPipeWrapper {
public:
   MediaPipeWrapper() {
    try {
        if (!Py_IsInitialized()) {
            py::initialize_interpreter();
        }

        py::exec(R"(
            import mediapipe as mp
            import numpy as np
            import cv2

            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_pose = mp.solutions.pose
            mp_hands = mp.solutions.hands

            # Custom connections to exclude face
            BODY_CONNECTIONS = frozenset([
                # Torso
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
                # Arms
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            ])

            # Initialize detectors
            pose = mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )

            hands = mp_hands.Hands(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                max_num_hands=2
            )

            def should_draw_connection(start_idx, end_idx, tracking_flags):
                # Define landmark ranges for each arm
                left_arm_landmarks = {
                    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    mp_pose.PoseLandmark.LEFT_ELBOW.value,
                    mp_pose.PoseLandmark.LEFT_WRIST.value
                }
                right_arm_landmarks = {
                    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                    mp_pose.PoseLandmark.RIGHT_WRIST.value
                }
                
                # Check if both landmarks belong to the same arm
                both_left = start_idx in left_arm_landmarks and end_idx in left_arm_landmarks
                both_right = start_idx in right_arm_landmarks and end_idx in right_arm_landmarks
                
                # Allow shoulder connection if either arm is active
                if (start_idx == mp_pose.PoseLandmark.LEFT_SHOULDER.value and 
                    end_idx == mp_pose.PoseLandmark.RIGHT_SHOULDER.value):
                    return tracking_flags['left_arm'] or tracking_flags['right_arm']
                
                # Check tracking flags
                if both_left:
                    return tracking_flags['left_arm']
                if both_right:
                    return tracking_flags['right_arm']
                
                # Default to drawing non-arm connections
                return True

            # Inside the draw_landmarks_no_face function in the Python code:
            def draw_landmarks_no_face(image, results, tracking_flags):
                annotated_image = image.copy()
                
                # Draw pose landmarks (excluding face)
                if results['pose_landmarks']:
                    # Draw custom connections (body only)
                    for connection in BODY_CONNECTIONS:
                        start_idx = connection[0].value
                        end_idx = connection[1].value
                        
                        # Skip if connection should not be drawn
                        if not should_draw_connection(start_idx, end_idx, tracking_flags):
                            continue
                        
                        start_landmark = results['pose_landmarks'].landmark[start_idx]
                        end_landmark = results['pose_landmarks'].landmark[end_idx]
                        
                        image_height, image_width = image.shape[:2]
                        start_point = (int(start_landmark.x * image_width), 
                                    int(start_landmark.y * image_height))
                        end_point = (int(end_landmark.x * image_width), 
                                int(end_landmark.y * image_height))
                        
                        cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)
                        cv2.circle(annotated_image, start_point, 4, (0, 0, 255), -1)
                        cv2.circle(annotated_image, end_point, 4, (0, 0, 255), -1)
                
                # Draw hand landmarks
                if results['hand_landmarks'] and results['multi_handedness']:
                    for hand_landmarks, handedness in zip(results['hand_landmarks'], 
                                                        results['multi_handedness']):
                        # Determine which hand this is
                        original_label = handedness.classification[0].label.lower()
                        hand_side = "right" if original_label == "left" else "left"
                        
                        # Only draw if both the arm is active AND fingers are toggled on
                        if tracking_flags[f'{hand_side}_arm'] and tracking_flags[f'{hand_side}_fingers']:
                            mp_drawing.draw_landmarks(
                                annotated_image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style()
                            )
                
                return annotated_image
        )");

        std::cout << "Successfully initialized MediaPipe detectors" << std::endl;

    } catch (const py::error_already_set& e) {
        std::cerr << "Python error in constructor: " << e.what() << std::endl;
        throw;
    }
}
    
    ~MediaPipeWrapper() {
        try {
            py::exec(R"(
                pose.close()
                hands.close()
            )");
        } catch (...) {}
    }
    
    // In MediaPipeWrapper::process_frame
    bool process_frame(const cv::Mat& frame,
                    Eigen::MatrixXd& pose_landmarks,
                    std::vector<Eigen::MatrixXd>& hand_landmarks,
                    cv::Mat& debug_output,
                    const std::map<std::string, bool>& activeArms,
                    const std::map<std::string, bool>& activeFingers) {
        try {
            // Convert BGR to RGB
            cv::Mat rgb_frame;
            cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
            
            // Convert to Python numpy array
            py::array_t<unsigned char> py_image(
                {rgb_frame.rows, rgb_frame.cols, 3},
                {rgb_frame.step[0], rgb_frame.step[1], rgb_frame.elemSize1()},
                rgb_frame.data
            );

            // Create Python dict for tracking flags
            py::dict tracking_flags;
            tracking_flags["left_arm"] = activeArms.at("left");
            tracking_flags["right_arm"] = activeArms.at("right");
            tracking_flags["left_fingers"] = activeFingers.at("left");
            tracking_flags["right_fingers"] = activeFingers.at("right");

            // Create the Python dictionary for the locals
            py::dict locals;
            locals["image_array"] = py_image;
            locals["tracking_flags"] = tracking_flags;

            // Process with MediaPipe
            py::exec(R"(
                try:
                    # Process frame
                    pose_results = pose.process(image_array)
                    hand_results = hands.process(image_array)

                    # Initialize results with empty landmarks if detection fails
                    results = {
                        'pose_landmarks': pose_results.pose_landmarks,
                        'hand_landmarks': hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else [],
                        'multi_handedness': hand_results.multi_handedness if hand_results.multi_handedness else []
                    }

                    # Always create a copy of the input image for debug output
                    debug_image = image_array.copy()

                    # Only draw landmarks if they were detected
                    if results['pose_landmarks'] or results['hand_landmarks']:
                        debug_image = draw_landmarks_no_face(image_array, results, tracking_flags)

                    # Initialize empty data structures for landmarks
                    pose_landmarks_data = []
                    hand_landmarks_data = []

                    # Get pose landmarks if available
                    if pose_results.pose_landmarks:
                        for landmark in pose_results.pose_landmarks.landmark:
                            pose_landmarks_data.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

                    # Get hand landmarks if available
                    if results['hand_landmarks']:
                        for hand_landmarks in results['hand_landmarks']:
                            hand_data = []
                            for landmark in hand_landmarks.landmark:
                                hand_data.append([landmark.x, landmark.y, landmark.z])
                            hand_landmarks_data.append(hand_data)

                except Exception as e:
                    print(f"Error in Python processing: {str(e)}")
                    debug_image = image_array.copy()  # Return original image if processing fails
                    pose_landmarks_data = []
                    hand_landmarks_data = []
            )", py::globals(), locals);

            // Get debug image (will be original frame if processing failed)
            py::array_t<unsigned char> debug_image = locals["debug_image"].cast<py::array_t<unsigned char>>();
            cv::Mat debug_mat(debug_image.shape(0), debug_image.shape(1), CV_8UC3, debug_image.mutable_data());
            cv::cvtColor(debug_mat, debug_output, cv::COLOR_RGB2BGR);

            // Get landmarks
            auto py_pose_landmarks = locals["pose_landmarks_data"].cast<py::list>();
            auto py_hand_landmarks = locals["hand_landmarks_data"].cast<py::list>();

            // Convert pose landmarks
            pose_landmarks.resize(py::len(py_pose_landmarks), 4);
            if (py::len(py_pose_landmarks) > 0) {
                for (size_t i = 0; i < py::len(py_pose_landmarks); ++i) {
                    auto landmark = py_pose_landmarks[i].cast<py::list>();
                    for (int j = 0; j < 4; ++j) {
                        pose_landmarks(i, j) = landmark[j].cast<double>();
                    }
                }
            }

            // Convert hand landmarks
            hand_landmarks.clear();
            for (auto hand_data : py_hand_landmarks) {
                auto hand_landmarks_list = hand_data.cast<py::list>();
                Eigen::MatrixXd hand_mat(py::len(hand_landmarks_list), 3);
                
                for (size_t i = 0; i < py::len(hand_landmarks_list); ++i) {
                    auto landmark = hand_landmarks_list[i].cast<py::list>();
                    for (int j = 0; j < 3; ++j) {
                        hand_mat(i, j) = landmark[j].cast<double>();
                    }
                }
                
                hand_landmarks.push_back(hand_mat);
            }

            // Return success even if no landmarks were detected
            return true;

        } catch (const py::error_already_set& e) {
            std::cerr << "Python error in process_frame: " << e.what() << std::endl;
            // Ensure debug_output is valid even on error
            frame.copyTo(debug_output);
            // Clear landmarks but don't throw
            pose_landmarks.resize(0, 4);
            hand_landmarks.clear();
            return true;  // Return true to keep the program running
        }
    }
};