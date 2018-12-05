// Created by Yu Wang on 03.12.18.
// The header file of Levenberg-Marquardt(LM) Optimizer.
// The LM Optimizer is defined as a class which need to be initialised by providing following params:
//  - lambda: float, damping factor (default: 0.001)
//  - max_iterations: Vector4i [4], number of max iterations for each pyramid level (default: [100, 100, 100, 100])
//  - precision: float (default: 5e-7)
//  - twist_init: Vector6f, initial value of twist coordinates (default: [0,0,0,0,0,0])

//

#ifndef RGBD_ODOMETRY_LM_OPTIMIZER_H
#define RGBD_ODOMETRY_LM_OPTIMIZER_H

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <vector>
#include <data_types.h>
#include <image_pyramid.h>
#include <lie_algebras.h>

namespace odometry
{

class LevenbergMarquardtOptimizer{
  public:
    // enable default constructor explicitly
    LevenbergMarquardtOptimizer();

    // parameterized constructor
    LevenbergMarquardtOptimizer(float lambda, float precision, const std::vector<int> kMaxIterations, const Vector6f& kTwistInit);

    // disable copy constructor
    LevenbergMarquardtOptimizer(const LevenbergMarquardtOptimizer& ) = delete;

    // disable copy assignment
    LevenbergMarquardtOptimizer& operator= ( const LevenbergMarquardtOptimizer & ) = delete;

    // solve the optimization, exposed to user
    Vector6f Solve(const ImagePyramid& kImagePyr1, const DepthPyramid& kDepthPyr1, const ImagePyramid& kImagePyr2);

    // for each new pair of consecutive frames, we need to reset the initial pose and lambda from user side, return status:
    // if -1: reset twist_init_ failed
    // otherwise: reset succeed
    OptimizerStatus SetInitialTwist(const Vector6f& kTwistInit);
    OptimizerStatus SetLambda(const float lambda);
    // reset accumulated statistics of the optimizer from user side after finish computing the camera pose:
    // number of iterations per pyramid level; energy values before/after each pyramid optimization
    void ResetStatistics();

  private:
    // the function that actually solves the optimization, return status:
    // if -1: failed, throw err, optimization terminate
    // otherwise: success
    OptimizerStatus OptimizeCameraPose(const ImagePyramid& kImagePyr1, const DepthPyramid& kDepthPyr1, const ImagePyramid& kImagePyr2, Vector6f& twist);

    // compute jacobians, weights, residuals and number of residuals, return status:
    // if -1: failed, throw err, compute terminate
    // otherwise: success
    OptimizerStatus ComputeResidualJacobian(const cv::Mat& kImg1, const cv::Mat& kImg2, const cv::Mat& kDep1, const Vector6f twist,
                                            Eigen::Matrix<float, Eigen::Dynamic, 6>& jaco,
                                            Eigen::DiagonalMatrix<float, Eigen::Dynamic, Eigen::Dynamic>& weight,
                                            Eigen::VectorXf& residual,
                                            int& num_residual);


    float lambda_; // will be modified during optimization, therefore need to be reset for the next pair of frames
    float precision_;
    std::vector<int> max_iterations_;
    Vector6f twist_init_; // need to be reset for the next pair of frames
    Vector6f twist_;  // always be {0} when constructed, the value changes as optimization progress
    std::vector<int> iters_stat_; // store the number of iterations performed per pyramid level
    std::vector<std::vector<float>> cost_stat_; // store the cost before/after optimization per pyramid level;
};


} // namespace odometry


#endif //RGBD_ODOMETRY_LM_OPTIMIZER_H
