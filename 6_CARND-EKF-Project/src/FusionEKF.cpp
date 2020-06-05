#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF()
{
    is_initialized_ = false;

    previous_timestamp_ = 0;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);
    Hj_ = MatrixXd(3, 4);
    Hj_ << 0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;
    H_laser_ << 1, 0, 0, 0,
        0, 1, 0, 0;
    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
        0, 0.0225;

    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;
    ekf_.F_ = MatrixXd(4, 4); // 状态转移矩阵
    ekf_.F_ << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;
    /**
   * TODO: Finish initializing the FusionEKF.
   * TODO: Set the process and measurement noises
   */
    // state covariance matrix P
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;
    // Initialize process noise covariance matrix
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ << 0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;
    // Initialize ekf state
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;
    noise_ax = 9; // 方差量，已经包含了平方项了
    noise_ay = 9;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack)
{
    /**
   * Initialization
   */
    if (!is_initialized_)
    {
        /**
     * TODO: Initialize the state ekf_.x_ with the first measurement.
     * TODO: Create the covariance matrix.
     * You'll need to convert radar from polar to cartesian coordinates.
     */

        // first measurement
        cout << "EKF: " << endl;
        // ekf_.x_ = VectorXd(4);
        // ekf_.x_ << 1, 1, 1, 1;
        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
        {
            // TODO: Convert radar from polar to cartesian coordinates
            //         and initialize state
            float rho = measurement_pack.raw_measurements_(0);
            float phi = measurement_pack.raw_measurements_(1);
            float rho_dot = measurement_pack.raw_measurements_(2);

            float px = rho * cos(phi);
            float py = rho * sin(phi);
            float vx = rho_dot * cos(phi);
            float vy = rho_dot * sin(phi);

            ekf_.x_ << px, py, vx, vy;
            // cout << "EKFRADARInit" << endl;
        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
        {
            // TODO: Initialize state.
            ekf_.x_ << measurement_pack.raw_measurements_(0),
                measurement_pack.raw_measurements_(1),
                0.0,
                0.0;
        }
        previous_timestamp_ = measurement_pack.timestamp_;
        // done initializing, no need to predict or update
        is_initialized_ = true;
        // cout << "EKFLidarInit" << endl;
        return;
    }
    /**
   * Prediction
   */

    /**
   * TODO: Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * TODO: Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
    // 两次测量之间的时间差

    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; //

    previous_timestamp_ = measurement_pack.timestamp_;
    // 状态转移矩阵
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;
    // 过程噪声矩阵
    float dt_2 = dt * dt;
    float dt_3 = dt_2 * dt;
    float dt_4 = dt_3 * dt;

    cout << "finish init666 ****************" << noise_ax << noise_ay <<  endl;
    ekf_.Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0,
        0, dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay,
        dt_3 / 2 * noise_ax, 0, dt_2 * noise_ax, 0,
        0, dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay;

    // float dt_2 = pow(dt, 2);
    // MatrixXd G(4, 2);
    // G << dt_2 / 2, 0,
    //     0, dt_2 / 2,
    //     dt, 0,
    //     0, dt;
    // MatrixXd Gt = G.transpose();
    // MatrixXd Qv(2, 2);
    // Qv << noise_ax, 0,
    //     0, noise_ay;
    // ekf_.Q_ = G * Qv;
    // ekf_.Q_ = ekf_.Q_ * Gt;

    // cout << ekf_.x_ << endl;
    // cout << "22here" << endl;
    ekf_.Predict();

    // cout << ekf_.x_ << endl;
    // cout << "224444here" << endl;
    /**
   * Update
   */

    /**
   * TODO:
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
    {
        // TODO: Radar updates
        ekf_.R_ = R_radar_;
        ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);
        // cout << ekf_.H_;
        // cout << ekf_.x_ << endl;
        // cout << "22444Radarere" << endl;
    }
    else
    {
        // TODO: Laser updates
        ekf_.R_ = R_laser_;
        ekf_.H_ = H_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
        // cout << ekf_.x_ << endl;
        // cout << "224444hLiadarere" << endl;
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}
