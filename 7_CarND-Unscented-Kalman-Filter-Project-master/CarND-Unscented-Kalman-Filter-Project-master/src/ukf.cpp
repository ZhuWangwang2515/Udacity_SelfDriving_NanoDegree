#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

/**
 * Initializes Unscented Kalman filter
 */
/*####################################################################################################
# Name             :
# Function         :
# Input Parameters :
# Return values    :
# Comments         :
####################################################################################################*/
UKF::UKF()
{
  is_initialized_ = false;
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.5;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values 
   */

  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  n_sigma_points_ = 2 * n_aug_ + 1;

  weights_ = VectorXd(n_sigma_points_);
  for (int i = 0; i < n_sigma_points_; i++)
  {
    weights_(i) = double(0.5 / (lambda_ + n_aug_));
  }
  weights_(0) = double(lambda_ / (lambda_ + n_aug_));

  Xsig_pred_ = MatrixXd(n_x_, n_sigma_points_);
  // Sensor's measurement size
  n_z_radar_ = 3; // radar -> rho, phi, rho_dot
  n_z_lidar_ = 2; // lidar -> px, py

  // Measurement covariance matrices
  R_lidar_ = MatrixXd(n_z_lidar_, n_z_lidar_);
  R_radar_ = MatrixXd(n_z_radar_, n_z_radar_);

  x_aug_ = VectorXd(n_aug_);
}

/*####################################################################################################
# Name             :
# Function         :
# Input Parameters :
# Return values    :
# Comments         :
####################################################################################################*/
UKF::~UKF() {}
/*####################################################################################################
# Name             :
# Function         :
# Input Parameters :
# Return values    :
# Comments         :
####################################################################################################*/
MatrixXd UKF::AugmentedSigmaPoints()
{
  // cout << "in augmented sigmapoints" << endl;
  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;

  // cout << "111" << endl;
  MatrixXd P_aug_ = MatrixXd(n_aug_, n_aug_);

  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1); // dimension of augmented sigma matrix

  MatrixXd Q = MatrixXd(2, 2);
  Q << std_a_ * std_a_, 0, 0, std_yawdd_ * std_yawdd_;
  P_aug_.fill(0);
  P_aug_.topLeftCorner(5, 5) = P_;
  P_aug_.bottomRightCorner(2, 2) = Q;

  MatrixXd A = P_aug_.llt().matrixL(); // llt matrix decomposition

  Xsig_aug_.col(0) = x_aug_;
  for (int i = 1; i <= n_aug_; i++)
  {
    Xsig_aug_.col(i) = x_aug_ + sqrt(lambda_ + n_aug_) * A.col(i - 1);
    Xsig_aug_.col(i + n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_) * A.col(i - 1);
  }
  return Xsig_aug_;
}
/*####################################################################################################
# Name             :
# Function         :
# Input Parameters :
# Return values    :
# Comments         :
####################################################################################################*/
MatrixXd UKF::SigmaPointPrediction(double const delta_t)
{
  // cout << "in sigmapointPrediction" << endl;
  MatrixXd Xsig_aug_ = AugmentedSigmaPoints();
  // cout << Xsig_aug_ << endl;
  // cout << "after augmented sigma points" << endl;
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // get parameters
    double px = Xsig_aug_(0, i);
    double py = Xsig_aug_(1, i);
    double v = Xsig_aug_(2, i);
    double yaw = Xsig_aug_(3, i);
    double yawd = Xsig_aug_(4, i);
    double nu_a = Xsig_aug_(5, i);
    double nu_yawdd = Xsig_aug_(6, i);

    if ((fabs(px) < 0.001) && (fabs(py) < 0.001))
    {
      px = 0.1;
      py = 0.1;
    }

    // define prediction paremeters
    double px_p, py_p, v_p, yaw_p, yawd_p;
    // predict sigma points
    // avoid division by zero
    if (fabs(yawd) > 0.001)
    {
      px_p = px + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = py + v / yawd * (-cos(yaw + yawd * delta_t) + cos(yaw));
    }
    else
    {
      px_p = px + v * cos(yaw) * delta_t;
      py_p = py + v * sin(yaw) * delta_t;
    }
    v_p = v;
    yaw_p = yaw + yawd * delta_t;
    yawd_p = yawd;

    // add error
    px_p += 0.5 * delta_t * delta_t * cos(yaw) * nu_a;
    py_p += 0.5 * delta_t * delta_t * sin(yaw) * nu_a;
    v_p += delta_t * nu_a;
    yaw_p += 0.5 * delta_t * delta_t * nu_yawdd;
    yawd_p += delta_t * nu_yawdd;

    // cout << "lalalalalal" << endl;
    // write predicted sigma points into right column
    // Xsig_pred_.col(i) << px_p, py_p, v_p, yaw_p, yawd_p;
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
    // cout << "l567890alalalalal" << endl;
  }
  return Xsig_pred_;
}
/*####################################################################################################
# Name             :
# Function         :
# Input Parameters :
# Return values    :
# Comments         :
####################################################################################################*/
void UKF::Prediction(double const delta_t)
{
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  // cout << "here3333333333333333" << endl;
  MatrixXd Xsig_pred = SigmaPointPrediction(delta_t);
  // VectorXd x = VectorXd(n_x_);
  x_.fill(0);
  // predict state mean
  // cout << "here2" << endl;
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // cout << "Iamhere" << endl;
    // cout << x_ << endl
    //  << Xsig_pred_.col(i) << endl;
    x_ = x_ + weights_(i) * Xsig_pred.col(i);
  }
  // MatrixXd P_ = MatrixXd(n_x_, n_x_);
  P_.fill(0);
  // predict state covariance matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd x_diff = Xsig_pred.col(i) - x_;
    // cout << x_diff << endl;
    // cout << "*****" << endl;

    // cout << x_diff(3) << endl;
    NormalizeAngle(x_diff(3));
    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
  // cout << "here3" << endl;
  // x_ = x;
  // P_ = P;
}

/*####################################################################################################
# Name             :
# Function         :
# Input Parameters :
# Return values    :
# Comments         :
####################################################################################################*/
void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  // set measurement dimension,radar can measure r,phi,and r_rot
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z_lidar_, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_lidar_);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    z_pred(0) += weights_(i) * Zsig(0, i);
    z_pred(1) += weights_(i) * Zsig(1, i);
  }

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_lidar_, n_z_lidar_);
  S.fill(0.0);
  // calculate innovation covariance matrix S
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd Zdiff = Zsig.col(i) - z_pred;
    NormalizeAngle(Zdiff(1));

    S += weights_(i) * (Zdiff * Zdiff.transpose());
  }

  R_lidar_.fill(0.0);
  R_lidar_(0, 0) = pow(std_laspx_, 2);
  R_lidar_(1, 1) = pow(std_laspy_, 2);

  S += R_lidar_;

  //**********
  //update state
  // Parse radar measurement
  VectorXd z = VectorXd(n_z_lidar_);
  z << meas_package.raw_measurements_[0],
      meas_package.raw_measurements_[1];
  // create matrix for cross correlation Tc, was used to calculate K gain
  MatrixXd Tc = MatrixXd(n_x_, n_z_lidar_);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd Zdiff = Zsig.col(i) - z_pred;
    // angle normalization
    NormalizeAngle(Zdiff(1));

    VectorXd Xdiff = Xsig_pred_.col(i) - x_;
    NormalizeAngle(Xdiff(3));
    Tc += weights_(i) * Xdiff * Zdiff.transpose();
  }

  // create kalman gain
  MatrixXd K = Tc * S.inverse();

  // update state mean and convariance matrix
  VectorXd Zdiff = z - z_pred;
  NormalizeAngle(Zdiff(1));

  x_ = x_ + K * Zdiff;
  P_ = P_ - K * S * K.transpose();
}

/*####################################################################################################
# Name             :
# Function         :
# Input Parameters :
# Return values    :
# Comments         :
+
####################################################################################################*/
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  //**************
  // prdict measurement
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_radar_, 2 * n_aug_ + 1);

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model
    Zsig(0, i) = sqrt(px * px + py * py);

    if (fabs(py) > 0.001 && fabs(px) > 0.001)
    {
      Zsig(1, i) = atan2(py, px);
    }
    else
    {
      Zsig(1, i) = 0.0;
    }

    if (fabs(sqrt(px * px + py * py)) > 0.001)
    {
      Zsig(2, i) = (px * v1 + py * v2) / sqrt(px * px + py * py);
    }
    else
    {
      Zsig(2, i) = 0.0;
    }
  }

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_radar_);
  // calculate mean predicted measurement
  z_pred.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_radar_, n_z_radar_);
  S.fill(0);
  // calculate innovation covariance matrix S
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd Zdiff = Zsig.col(i) - z_pred;
    NormalizeAngle(Zdiff(1));

    S += weights_(i) * (Zdiff * Zdiff.transpose());
  }
  R_radar_.fill(0);
  R_radar_(0, 0) = pow(std_radr_, 2);
  R_radar_(1, 1) = pow(std_radphi_, 2);
  R_radar_(2, 2) = pow(std_radrd_, 2);
  S += R_radar_;

  //**********
  //update state
  // Parse radar measurement
  VectorXd z = VectorXd(n_z_radar_);
  z << meas_package.raw_measurements_[0],
      meas_package.raw_measurements_[1],
      meas_package.raw_measurements_[2];
  // create matrix for cross correlation Tc, was used to calculate K gain
  MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd Zdiff = Zsig.col(i) - z_pred;
    // angle normalization
    NormalizeAngle(Zdiff(1));

    VectorXd Xdiff = Xsig_pred_.col(i) - x_;
    NormalizeAngle(Xdiff(3));
    Tc += weights_(i) * Xdiff * Zdiff.transpose();
  }

  // create kalman gain
  MatrixXd K = Tc * S.inverse();

  // update state mean and convariance matrix
  VectorXd Zdiff = z - z_pred;
  NormalizeAngle(Zdiff(1));

  x_ = x_ + K * Zdiff;
  P_ = P_ - K * S * K.transpose();
}
/*####################################################################################################
# Name             :
# Function         :
# Input Parameters :
# Return values    :
# Comments         :
####################################################################################################*/
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
  cout << "running" << endl;
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (!is_initialized_)
  {
    cout << "UKF" << endl;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {

      // cout << "here" << endl;
      float rho = meas_package.raw_measurements_(0);
      float phi = meas_package.raw_measurements_(1);
      float rho_dot = meas_package.raw_measurements_(2);

      // cout << "here4" << endl;
      float px = rho * cos(phi);
      float py = rho * sin(phi);

      // cout << "here6" << endl;
      x_ << px, py, rho_dot, 0.0, 0.0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {

      // cout << "here2" << endl;
      x_ << meas_package.raw_measurements_(0),
          meas_package.raw_measurements_(1),
          0.0,
          0.0,
          0.0;

      //
    }
    time_us_ = meas_package.timestamp_;

    // Prediction(time_us_);
    // if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    // {
    //   UpdateRadar(meas_package);
    // }

    // else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    // {
    //   UpdateLidar(meas_package);
    // }

    is_initialized_ = true;

    return;
  }

  /*****************************************
   * Prediction
   ****************************************/
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  // cout << "here222" << endl;
  Prediction(dt);
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(meas_package);
  }

  else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    UpdateLidar(meas_package);
  }
}
/*####################################################################################################
# Name             :
# Function         :
# Input Parameters :
# Return values    :
# Comments         :
####################################################################################################*/

void UKF::NormalizeAngle(double &phi)
{
  while (phi > M_PI)
    phi -= 2 * M_PI;
  while (phi < -M_PI)
    phi += 2 * M_PI;
}