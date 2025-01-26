#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "arm_tracker.hpp"

namespace py = pybind11;

PYBIND11_MODULE(arm_tracker_python, m) {
    py::class_<ArmTracker>(m, "ArmTracker")
        .def(py::init<>())
        .def("process_frame", [](ArmTracker& self, const py::array_t<uint8_t>& frame,
                                const py::array_t<double>& pose_data,
                                const std::vector<py::array_t<double>>& hand_data) {
            // Convert numpy array to cv::Mat
            auto frame_info = frame.request();
            cv::Mat cv_frame(frame_info.shape[0], frame_info.shape[1], 
                        CV_8UC3, (void*)frame_info.ptr);
            
            // Process landmarks
            ArmTracker::TrackingResult result;
            self.processFrameWithLandmarks(cv_frame, pose_data, hand_data, result);
            
            // Convert result to Python dict
            py::dict py_result;
            py_result["tracking_lost"] = result.trackingLost;
            // Add other result data...
            
            return py_result;
        })
        .def("toggle_arm", &ArmTracker::toggleArm)
        .def("toggle_fingers", &ArmTracker::toggleFingers);
}
