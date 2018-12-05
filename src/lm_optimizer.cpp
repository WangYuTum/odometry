// Created by Yu Wang on 03.12.18.
// Implementation of LM Optimizer class.

#include <lm_optimizer.h>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <image_processing_global.h>

namespace odometry
{

LevenbergMarquardtOptimizer::LevenbergMarquardtOptimizer() {
  lambda_ = 0.001;
  precision_ = 5e-7;
  twist_init_ << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  twist_ = twist_init_;
  for (int i = 0; i < 4; i++){
    max_iterations_.push_back(100);
    iters_stat_.push_back(0);
    cost_stat_.push_back(std::vector<float>{0.0, 0.0});
  }
}

LevenbergMarquardtOptimizer::LevenbergMarquardtOptimizer(float lambda,
                                                         float precision,
                                                         const std::vector<int> kMaxIterations,
                                                         const Vector6f& kTwistInit){
  lambda_ = lambda;
  precision_ = precision;
  max_iterations_ = kMaxIterations;
  twist_init_ = kTwistInit;
  twist_ << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  for (int i = 0; i < 4; i++){
    iters_stat_.push_back(0);
    cost_stat_.push_back(std::vector<float>{0.0, 0.0});
  }
}

Vector6f LevenbergMarquardtOptimizer::Solve(const ImagePyramid& kImagePyr1,
                                            const DepthPyramid& kDepthPyr1,
                                            const ImagePyramid& kImagePyr2){
  OptimizerStatus status;
  status = OptimizeCameraPose(kImagePyr1, kDepthPyr1, kImagePyr2, twist_);
  if (status == -1) {
    std::cout << "Optimize failed! " << std::endl;
    Vector6f tmp;
    tmp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    return tmp;
  }
  else{
    return twist_;
  }
}

void LevenbergMarquardtOptimizer::ResetStatistics(){
  // set all statistics to zeros
  for (int i = 0; i < 4; i++){
    iters_stat_[i] = 0;
    cost_stat_[i] = {0.0, 0.0};
  }
}

// The function that actually solves the optimization
OptimizerStatus LevenbergMarquardtOptimizer::OptimizeCameraPose(const ImagePyramid& kImagePyr1,
                                   const DepthPyramid& kDepthPyr1,
                                   const ImagePyramid& kImagePyr2,
                                   Vector6f& twist){
  twist = twist_init_; // initial pose
  Vector6f increment_twist; // initial increment twist
  increment_twist << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  int pyr_levels = kImagePyr1.GetNumberLevels();
  int l = pyr_levels-1;
  // loop for each pyramid level
  while (l >= 0){
    // get respective images/depth map from current pyramid level as const reference
    const cv::Mat& kImg1 = kImagePyr1.GetPyramidImage(l); // cv::CV_8U, 0-255
    const cv::Mat& kImg2 = kImagePyr2.GetPyramidImage(l); // cv::CV_8U, 0-255
    const cv::Mat& kDep1 = kDepthPyr1.GetPyramidDepth(l); // cv::CV_32FC1
    // check data types and matrix size
    if ((kImg1.rows != kImg2.rows) || (kImg1.rows != kImg2.rows) || (kImg1.rows != kDep1.rows)){
      std::cout << "Image rows don't match in LevenbergMarquardtOptimizer::OptimizeCameraPose()." << std::endl;
      return -1;
    }
    if ((kImg1.cols != kImg2.cols) || (kImg1.cols != kImg2.cols) || (kImg1.cols != kDep1.cols)){
      std::cout << "Image cols don't match in LevenbergMarquardtOptimizer::OptimizeCameraPose()." << std::endl;
      return -1;
    }
    if ((kImg1.type() != PixelType) || (kImg2.type() != PixelType) || (kDep1.type() != PixelType)){
      std::cout << "Image types don't match in LevenbergMarquardtOptimizer::OptimizeCameraPose()." << std::endl;
      return -1;
    }
    int iter_count = 0;
    float err_last = 1e+25;
    float err_now = 0.0;
    while ((err_last - err_now) > precision_ && max_iterations_[l]> iter_count){
      // declare Jacobian, Weight, Residual and num_residual since we don't know the size of them,
      // we need to declare them at each iteration
      Eigen::Matrix<float, Eigen::Dynamic, 6> jaco;
      Eigen::DiagonalMatrix<float, Eigen::Dynamic, Eigen::Dynamic> weights;
      Eigen::VectorXf residuals;
      int num_residuals = 0;
      Vector6f incremented_twist; // TODO: get incremented twist
      OptimizerStatus compute_status = ComputeResidualJacobian(kImg1, kImg2, kDep1, incremented_twist, jaco, weights, residuals, num_residuals);
      if (compute_status == -1){
        std::cout << "Evaluate Residual & Jacobian failed " << std::endl;
        return -1;
      }
      // compute jacobian succeed, proceed
      err_now = (1.0 / float(num_residuals)) * residuals.transpose() * weights * residuals;
      if (err_now > err_last){
        lambda_ = 10.0 * lambda_;
        // increment_twist = 0.0; TODO: solve the linear system
      } else{
        // twist = 0.0; TODO: update current pose estimate with incremented_twist
        err_last = err_now;
        lambda_ = lambda_ / 10.0;
        // increment_twist = 0.0;  TODO: solve the linear system
      }
      iter_count++;
    } // end optimize criteria loop
    l--;
  } // end pyramid loop
}

OptimizerStatus LevenbergMarquardtOptimizer::ComputeResidualJacobian(const cv::Mat& kImg1,
                                                                     const cv::Mat& kImg2,
                                                                     const cv::Mat& kDep1,
                                                                     const Vector6f twist,
                                                                     Eigen::Matrix<float, Eigen::Dynamic, 6>& jaco,
                                                                     Eigen::DiagonalMatrix<float, Eigen::Dynamic, Eigen::Dynamic>& weight,
                                                                     Eigen::VectorXf& residual,
                                                                     int& num_residual){
  int kRows = kImg1.rows;
  int kCols = kImg1.cols;
  cv::Mat warped_img(kRows, kCols, CV_8U);
  // note that not all pixels in the warped image have valid intensity since some will be warped outside the image boundary
  WarpImage(kImg2, twist, warped_img);

  // need to check the following before return, if any check is failed, return unsuccessful
  // num_residual != 0
  // residual != (inf || nan)
  // weight != (inf || nan)
  // jaco != (inf || nan)
  return 0;
}

OptimizerStatus LevenbergMarquardtOptimizer::SetInitialTwist(const Vector6f& kTwistInit){
  twist_init_ = kTwistInit;
  // return -1;
  return 0;
}

OptimizerStatus LevenbergMarquardtOptimizer::SetLambda(const float lambda = 0.001){
  lambda_ = lambda;
  // return -1;
  return 0;
}

} // namespace odometry

