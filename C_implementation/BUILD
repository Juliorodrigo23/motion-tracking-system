load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "arm_tracker",
    srcs = [
        "src/main.cpp",
        "src/arm_tracker.cpp",
    ],
    hdrs = [
        "include/arm_tracker.hpp",
    ],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:landmark_cc_proto",
        "@mediapipe//mediapipe/modules/hand_landmark:hand_landmark_tracking_cpu",
        "@mediapipe//mediapipe/modules/pose_landmark:pose_landmark_cpu",
        "@opencv//:opencv",
    ],
    includes = ["include"],
    copts = ["-std=c++17"],
)