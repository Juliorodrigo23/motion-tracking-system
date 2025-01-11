#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

namespace py = pybind11;

class MediaPipeWrapper {
public:
    MediaPipeWrapper() {
        // Initialize Python
        py::initialize_interpreter();
        
        // Import MediaPipe
        auto mediapipe = py::module::import("mediapipe.python");
        auto solutions = py::module::import("mediapipe.solutions");
        
        // Create pose and hands objects
        mp_pose = solutions.attr("pose").attr("Pose")(
            py::arg("min_detection_confidence")=0.5,
            py::arg("min_tracking_confidence")=0.5,
            py::arg("model_complexity")=1
        );
        
        mp_hands = solutions.attr("hands").attr("Hands")(
            py::arg("min_detection_confidence")=0.5,
            py::arg("min_tracking_confidence")=0.5,
            py::arg("max_num_hands")=2
        );
    }
    
    ~MediaPipeWrapper() {
        // Release Python objects
        mp_pose = py::none();
        mp_hands = py::none();
        // Finalize Python
        py::finalize_interpreter();
    }
    
    bool process_frame(const cv::Mat& frame,
                      Eigen::MatrixXd& pose_landmarks,
                      std::vector<Eigen::MatrixXd>& hand_landmarks) {
        // Convert BGR to RGB
        cv::Mat rgb_frame;
        cv::cvtColor(frame, cv::COLOR_BGR2RGB, rgb_frame);
        
        // Convert to Python numpy array
        py::array_t<unsigned char> py_image(
            {frame.rows, frame.cols, 3},
            {frame.step[0], frame.step[1], frame.elemSize1()},
            frame.data
        );
        
        // Process with MediaPipe
        auto pose_results = mp_pose.attr("process")(py_image);
        auto hands_results = mp_hands.attr("process")(py_image);
        
        // Extract pose landmarks
        if (!pose_results.attr("pose_landmarks").is_none()) {
            auto landmarks = pose_results.attr("pose_landmarks").attr("landmark");
            int num_landmarks = py::len(landmarks);
            pose_landmarks.resize(num_landmarks, 4);
            
            for (int i = 0; i < num_landmarks; ++i) {
                auto lm = landmarks[i];
                pose_landmarks(i, 0) = lm.attr("x").cast<double>();
                pose_landmarks(i, 1) = lm.attr("y").cast<double>();
                pose_landmarks(i, 2) = lm.attr("z").cast<double>();
                pose_landmarks(i, 3) = lm.attr("visibility").cast<double>();
            }
        } else {
            pose_landmarks.resize(0, 4);
            return false;
        }
        
        // Extract hand landmarks
        hand_landmarks.clear();
        if (!hands_results.attr("multi_hand_landmarks").is_none()) {
            auto multi_hand_landmarks = hands_results.attr("multi_hand_landmarks");
            for (auto hand_landmarks_obj : multi_hand_landmarks) {
                auto landmarks = hand_landmarks_obj.attr("landmark");
                int num_landmarks = py::len(landmarks);
                Eigen::MatrixXd hand_mat(num_landmarks, 3);
                
                for (int i = 0; i < num_landmarks; ++i) {
                    auto lm = landmarks[i];
                    hand_mat(i, 0) = lm.attr("x").cast<double>();
                    hand_mat(i, 1) = lm.attr("y").cast<double>();
                    hand_mat(i, 2) = lm.attr("z").cast<double>();
                }
                
                hand_landmarks.push_back(hand_mat);
            }
        }
        
        return true;
    }
    
private:
    py::object mp_pose;
    py::object mp_hands;
};
