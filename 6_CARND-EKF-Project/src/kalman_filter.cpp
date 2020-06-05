#include "kalman_filter.h"
#include <Eigen/Dense>
#include <stdio.h>
#include <iostream>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in)
{
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict()
{
  /**
   * TODO: predict the state
   */
  x_ = F_ * x_;

  // cout << "Predictin" << endl;
  MatrixXd Ft = F_.transpose();
  // cout << "Predictout" << endl;
  P_ = F_ * P_ * Ft + Q_;
  // cout << "Predictout222" << endl;
}
void KalmanFilter::UpdateEKF(const VectorXd &z)
{
  /**
   * TODO: update the state by using Kalman Filter equations
   */
  // 预测过程
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  // Map predicted state into measurement space
  double rho = sqrt(px * px + py * py);
  double phi = atan2(py, px);
  double rho_dot = (px * vx + py * vy) / std::max(rho, 0.0001);

  VectorXd z_pred(3);
  z_pred << rho, phi, rho_dot;

  VectorXd y = z - z_pred;

  // VectorXd z_old(3);

  // double rho = sqrt(pow(x_(0), 2) + pow(x_(1), 2));
  // double phi = atan2(x_(2), x_(1));
  // double rho_dot = (x_(0) * x_(2) + x_(1) * x_(3)) / std::max(rho, 0.00001);

  // z_old << rho, phi, rho_dot;

  // VectorXd y = z - z_old;

  // 将角度控制在-180-180度之间
  while (y(1) > 3.14159)
    y(1) -= 2 * 3.14159;
  while (y(1) < -3.14159)
    y(1) += 2 * 3.14159;

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  cout << S << endl;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;
  // new state
  // cout << "inupd11111ateEkf" << endl;
  // cout << x_ << endl;
  // cout << P_ << endl;
  // cout << Ht << endl;
  // cout << Si << endl;
  // cout << y << endl;
  x_ = x_ + K * y;
  // cout << "inupdateEkf" << endl;
  // cout << x_ << endl;
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z)
{
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */
  VectorXd y = z - H_ * x_;

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;
  // new state
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
