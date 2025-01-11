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

            // Draw utils will be needed for visualization
            py::exec(R"(
                import mediapipe as mp
                import numpy as np

                # Initialize MediaPipe drawing utils
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                mp_pose = mp.solutions.pose
                mp_hands = mp.solutions.hands

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

                def draw_landmarks(image, results):
                    # Draw pose landmarks
                    if results['pose_landmarks']:
                        mp_drawing.draw_landmarks(
                            image,
                            results['pose_landmarks'],
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                    
                    # Draw hand landmarks
                    if results['hand_landmarks']:
                        for hand_landmarks in results['hand_landmarks']:
                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style()
                            )
                    
                    return image
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
                      std::vector<Eigen::MatrixXd>& hand_landmarks) {
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

            // Process with MediaPipe
            py::dict locals;
            locals["image_array"] = py_image;

            py::exec(R"(
                # Process frame
                pose_results = pose.process(image_array)
                hand_results = hands.process(image_array)

                # Convert results for debugging visualization
                results = {
                    'pose_landmarks': pose_results.pose_landmarks,
                    'hand_landmarks': hand_results.multi_hand_landmarks
                }

                # Draw landmarks on debug image
                debug_image = image_array.copy()
                debug_image = draw_landmarks(debug_image, results)

                # Get pose landmarks
                pose_landmarks_data = []
                if pose_results.pose_landmarks:
                    for landmark in pose_results.pose_landmarks.landmark:
                        pose_landmarks_data.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                    print(f"Found {len(pose_landmarks_data)} pose landmarks")

                # Get hand landmarks
                hand_landmarks_data = []
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        hand_data = []
                        for landmark in hand_landmarks.landmark:
                            hand_data.append([landmark.x, landmark.y, landmark.z])
                        hand_landmarks_data.append(hand_data)
                    print(f"Found {len(hand_landmarks_data)} hands")
            )", py::globals(), locals);

            // Get debug image and display it
            py::array_t<unsigned char> debug_image = locals["debug_image"].cast<py::array_t<unsigned char>>();
            cv::Mat debug_mat(debug_image.shape(0), debug_image.shape(1), CV_8UC3, debug_image.mutable_data());
            cv::cvtColor(debug_mat, debug_mat, cv::COLOR_RGB2BGR);
            cv::imshow("MediaPipe Debug", debug_mat);

            // Get landmarks
            auto py_pose_landmarks = locals["pose_landmarks_data"].cast<py::list>();
            auto py_hand_landmarks = locals["hand_landmarks_data"].cast<py::list>();

            // Convert pose landmarks to Eigen matrix
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

            // Convert hand landmarks to vector of Eigen matrices
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