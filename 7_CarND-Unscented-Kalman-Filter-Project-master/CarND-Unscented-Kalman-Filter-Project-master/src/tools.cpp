#include "tools.h"

using Eigen::VectorXd;
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