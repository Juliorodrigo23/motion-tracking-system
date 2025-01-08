#pragma once
#include <Eigen/Dense>

class JointKalmanFilter {
public:
    JointKalmanFilter() {
        // State: [x, y, z, vx, vy, vz]
        A = Eigen::MatrixXd::Identity(6, 6);
        A.block<3,3>(0,3) = Eigen::MatrixXd::Identity(3,3) * dt;
        
        // Measurement matrix (only position is measured)
        H = Eigen::MatrixXd::Zero(3, 6);
        H.block<3,3>(0,0) = Eigen::MatrixXd::Identity(3,3);
        
        // Process noise
        Q = Eigen::MatrixXd::Identity(6, 6);
        Q.block<3,3>(0,0) *= 0.1;  // Position noise
        Q.block<3,3>(3,3) *= 0.2;  // Velocity noise
        
        // Measurement noise
        R = Eigen::MatrixXd::Identity(3, 3) * 0.1;
        
        // Initial state covariance
        P = Eigen::MatrixXd::Identity(6, 6) * 1.0;
        
        // Initial state
        x = Eigen::VectorXd::Zero(6);
    }
    
    void predict() {
        x = A * x;
        P = A * P * A.transpose() + Q;
    }
    
    void update(const Eigen::Vector3d& measurement) {
        Eigen::VectorXd z(3);
        z << measurement;
        
        Eigen::MatrixXd K = P * H.transpose() * (H * P * H.transpose() + R).inverse();
        x = x + K * (z - H * x);
        P = (Eigen::MatrixXd::Identity(6, 6) - K * H) * P;
    }
    
    Eigen::Vector3d getPosition() const {
        return x.head(3);
    }
    
    Eigen::Vector3d getVelocity() const {
        return x.tail(3);
    }

private:
    static constexpr double dt = 1.0/30.0;  // Assuming 30 FPS
    
    Eigen::MatrixXd A;  // State transition matrix
    Eigen::MatrixXd H;  // Measurement matrix
    Eigen::MatrixXd Q;  // Process noise covariance
    Eigen::MatrixXd R;  // Measurement noise covariance
    Eigen::MatrixXd P;  // Error covariance matrix
    Eigen::VectorXd x;  // State vector
};