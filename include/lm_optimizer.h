// Created by Yu Wang on 03.12.18.
// The header file of Levenberg-Marquardt(LM) Optimizer.
// The LM Optimizer is defined as a class which need to be initialised by providing following params:
//  - lambda: double, damping factor (default: 0.001)
//  - max_iterations: Vector4i [4], number of max iterations for each pyramid level (default: [100, 100, 100, 100])
//  - precision: double (default: 5e-7)
//  - twist_init: Vector6d, initial value of twist coordinates (default: [0,0,0,0,0,0])

//

#ifndef RGBD_ODOMETRY_LM_OPTIMIZER_H
#define RGBD_ODOMETRY_LM_OPTIMIZER_H

#include "data_types.h"
#include "image_pyramid.h"

namespace odometry
{

class LevenbergMarquardtOptimizer{
  public:
    LevenbergMarquardtOptimizer() = default;
    LevenbergMarquardtOptimizer(double lambda, double precision, const Vector4i& kMaxIterations, const Vector6d& kTwistInit);
    LevenbergMarquardtOptimizer(const LevenbergMarquardtOptimizer& ) = delete;
    LevenbergMarquardtOptimizer & LevenbergMarquardtOptimizer :: operator= ( const LevenbergMarquardtOptimizer & ) = delete;
    // solve the optimization, return final pose as value
    Vector6d Solve(const ImagePyramid& kImagePyr1, const DepthPyramid& kDepthPyr1, const ImagePyramid& kImagePyr2);

  private:
    // the function that actually solves the optimization, return status:
    // if return -1: failed, throw err, optimization terminate
    // if return 0: success
    OptimizerStatus OptimizeCameraPose(const ImagePyramid& kImagePyr1, const DepthPyramid& kDepthPyr1, const ImagePyramid& kImagePyr2, Vector6d& twist);

    double lambda_;
    double precision_;
    Vector4i max_iterations_;
    Vector6d twist_init_;
    Vector6d twist_;
};


} // namespace odometry


#endif //RGBD_ODOMETRY_LM_OPTIMIZER_H
