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
#include <camera.h>

namespace odometry
{

class LevenbergMarquardtOptimizer{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // disable default constructor explicitly
    LevenbergMarquardtOptimizer() = delete;

    // parameterized constructor
    LevenbergMarquardtOptimizer(float lambda, float precision, const std::vector<int> kMaxIterations, const Matrix44f& kTwistInit, const std::shared_ptr<CameraPyramid>& kCameraPtr);

    // destructor to handle pointers & dynamic memory
    ~ LevenbergMarquardtOptimizer();

    // disable copy constructor
    LevenbergMarquardtOptimizer(const LevenbergMarquardtOptimizer& ) = delete;

    // disable copy assignment
    LevenbergMarquardtOptimizer& operator= ( const LevenbergMarquardtOptimizer & ) = delete;

    // solve the optimization, exposed to user
    Matrix44f Solve(const ImagePyramid& kImagePyr1, const DepthPyramid& kDepthPyr1, const ImagePyramid& kImagePyr2);

    // show statistics
    void ShowReport();

    // reset optimizer after computing each pair of frames, need to be called from user's side. the method does the following:
    // - reset initial pose: 4x4 float matrix
    // - reset damping factor: float
    // - clear all statistics: iters_stat_, cost_stat_
    // return -1 if reset failed, otherwise success
    OptimizerStatus Reset(const Matrix44f& kTwistInit, const float lambda);

  private:
    // the function that actually solves the optimization, return status:
    // if -1: failed, throw err, optimization terminate
    // otherwise: success
    OptimizerStatus OptimizeCameraPose(const ImagePyramid& kImagePyr1, const DepthPyramid& kDepthPyr1, const ImagePyramid& kImagePyr2);

    // compute jacobians, weights, residuals and number of residuals, return status:
    // if -1: failed, throw err, compute terminate
    // otherwise: success
    // Naive impl, big loop over all pixels with openmp
    OptimizerStatus ComputeResidualJacobianNaive(const cv::Mat& kImg1, const cv::Mat& kImg2, const cv::Mat& kDep1, const Matrix44f& twist,
                                                  Eigen::Matrix<float, Eigen::Dynamic, 6>& jaco,
                                                  Eigen::DiagonalMatrix<float, Eigen::Dynamic, Eigen::Dynamic>& weight,
                                                  Eigen::VectorXf& residual,
                                                  int& num_residual,
                                                  int level);
    // SSE impl, highly optimized
    OptimizerStatus ComputeResidualJacobianSse(const cv::Mat& kImg1, const cv::Mat& kImg2, const cv::Mat& kDep1, const Matrix44f& twist,
                                                Eigen::Matrix<float, Eigen::Dynamic, 6>& jaco,
                                                Eigen::DiagonalMatrix<float, Eigen::Dynamic, Eigen::Dynamic>& weight,
                                                Eigen::VectorXf& residual,
                                                int& num_residual);


    void SetIdentityTransform(Matrix44f& in_mat);

    OptimizerStatus SetInitialTwist(const Matrix44f& kTwistInit);
    OptimizerStatus SetLambda(const float lambda);
    // reset accumulated statistics of the optimizer after finish computing the camera pose:
    // number of iterations per pyramid level; energy values before/after each pyramid optimization
    OptimizerStatus ResetStatistics();


    /************************************** PRIVATE DATA MEMBERS ********************************************/
    float lambda_; // will be modified during optimization, therefore need to be reset for the next pair of frames
    float precision_;
    std::vector<int> max_iterations_;
    Matrix44f twist_init_; // need to be reset for the next pair of frames
    Matrix44f twist_;  // always be identity when constructed, the value is changed after optimization
    std::vector<int> iters_stat_; // store the number of iterations performed per pyramid level
    std::vector<std::vector<float>> cost_stat_; // store the cost before/after optimization per pyramid level;

    // shared pointer to a camera. note that the pointer MUST point to one global camera instance
    // during the entire lifetime of the program
    std::shared_ptr<CameraPyramid> camera_ptr_;
};


} // namespace odometry


#endif //RGBD_ODOMETRY_LM_OPTIMIZER_H
