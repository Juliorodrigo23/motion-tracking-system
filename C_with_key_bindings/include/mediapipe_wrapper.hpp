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
                import cv2  # Added cv2 import

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

                def draw_landmarks_no_face(image, results):
                    annotated_image = image.copy()
                    
                    # Draw pose landmarks (excluding face)
                    if results['pose_landmarks']:
                        # Draw custom connections (body only)
                        for connection in BODY_CONNECTIONS:
                            start_idx = connection[0].value
                            end_idx = connection[1].value
                            
                            start_landmark = results['pose_landmarks'].landmark[start_idx]
                            end_landmark = results['pose_landmarks'].landmark[end_idx]
                            
                            image_height, image_width = image.shape[:2]
                            start_point = (int(start_landmark.x * image_width), int(start_landmark.y * image_height))
                            end_point = (int(end_landmark.x * image_width), int(end_landmark.y * image_height))
                            
                            cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)
                            cv2.circle(annotated_image, start_point, 4, (0, 0, 255), -1)
                            cv2.circle(annotated_image, end_point, 4, (0, 0, 255), -1)
                    
                    # Draw hand landmarks
                    if results['hand_landmarks']:
                        for hand_landmarks in results['hand_landmarks']:
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
    
    bool process_frame(const cv::Mat& frame,
                      Eigen::MatrixXd& pose_landmarks,
                      std::vector<Eigen::MatrixXd>& hand_landmarks,
                      cv::Mat& debug_output) {
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

            // Create the Python dictionary for the locals
            py::dict locals;
            locals["image_array"] = py_image;

            // Process with MediaPipe
            py::exec(R"(
                # Process frame
                pose_results = pose.process(image_array)
                hand_results = hands.process(image_array)

                results = {
                    'pose_landmarks': pose_results.pose_landmarks,
                    'hand_landmarks': hand_results.multi_hand_landmarks
                }

                # Draw landmarks
                debug_image = draw_landmarks_no_face(image_array, results)

                # Get pose landmarks
                pose_landmarks_data = []
                if pose_results.pose_landmarks:
                    for landmark in pose_results.pose_landmarks.landmark:
                        pose_landmarks_data.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

                # Get hand landmarks
                hand_landmarks_data = []
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        hand_data = []
                        for landmark in hand_landmarks.landmark:
                            hand_data.append([landmark.x, landmark.y, landmark.z])
                        hand_landmarks_data.append(hand_data)
            )", py::globals(), locals);

            // Get debug image
            py::array_t<unsigned char> debug_image = locals["debug_image"].cast<py::array_t<unsigned char>>();
            cv::Mat debug_mat(debug_image.shape(0), debug_image.shape(1), CV_8UC3, debug_image.mutable_data());
            cv::cvtColor(debug_mat, debug_output, cv::COLOR_RGB2BGR);

            // Get landmarks
            auto py_pose_landmarks = locals["pose_landmarks_data"].cast<py::list>();
            auto py_hand_landmarks = locals["hand_landmarks_data"].cast<py::list>();

            // Convert pose landmarks
            if (py::len(py_pose_landmarks) > 0) {
                pose_landmarks.resize(py::len(py_pose_landmarks), 4);
                for (size_t i = 0; i < py::len(py_pose_landmarks); ++i) {
                    auto landmark = py_pose_landmarks[i].cast<py::list>();
                    for (int j = 0; j < 4; ++j) {
                        pose_landmarks(i, j) = landmark[j].cast<double>();
                    }
                }
            } else {
                pose_landmarks.resize(0, 4);
                return false;
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

            return true;
        } catch (const py::error_already_set& e) {
            std::cerr << "Python error in process_frame: " << e.what() << std::endl;
            return false;
        }
    }
};