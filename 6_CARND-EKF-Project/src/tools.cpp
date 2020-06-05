#include "tools.h"
#include <iostream>
#include <Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
   /**
   * TODO: Calculate the RMSE here.
   */
   VectorXd rmse(4);
   rmse << 0, 0, 0, 0;
   // 要求测量值不能为空
   if (estimations.size() == 0)
      // 要求测量值不能为空
      if (estimations.size() == 0)
      {
         return rmse;
      }
   // 要求测量值的数量和真值的数量相同
   if (estimations.size() != ground_truth.size())
   {
      return rmse;
   }
   // 迭代求均方根
   for (int i = 0; i < estimations.size(); i++)
   {
      VectorXd residual = estimations[i].array() - ground_truth[i].array();
      residual = residual.array() * residual.array();
      rmse += residual;
   }
   rmse /= estimations.size();
   rmse = rmse.array().sqrt();
   return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state)
{
   /**
   * TODO:
   * Calculate a Jacobian here.
   */
   MatrixXd Hj(3, 4);
   // recover state parameters
   float px = x_state(0);
   float py = x_state(1);
   float vx = x_state(2);
   float vy = x_state(3);

   // TODO: YOUR CODE HERE

   // check division by zero

   // compute the Jacobian matrix
   float pxpy = px * px + py * py;
   float pxpy_1_2 = sqrt(pxpy);
   float pxpy_3_2 = pxpy * pxpy_1_2;
   cout << "Pxpy" << pxpy  << '\t' << pxpy_1_2 << '\t' << pxpy_3_2 << endl;
   if (std::abs(pxpy) < 0.0001)
   {
      cout << "Calicatution error" << endl;
      return Hj;
   }
   Hj << (px / pxpy_1_2), (py / pxpy_1_2), 0, 0,
       (-py / pxpy), (px / pxpy), 0, 0,
       py * (vx * py - vy * px) / pxpy_3_2, px * (vy * px - vx * py) / pxpy_3_2, px / pxpy_1_2, py / pxpy_1_2;
   return Hj;
}
