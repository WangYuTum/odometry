// Created by Yu Wang on 03.12.18.
// Implementation of LM Optimizer class.

#include <lm_optimizer.h>
#include <iostream>

namespace odometry
{

// TODO: check pass by reference copy values
LevenbergMarquardtOptimizer::LevenbergMarquardtOptimizer(double lambda,
                                                         double precision,
                                                         const Vector4i& kMaxIterations,
                                                         const Vector6d& kTwistInit){
  lambda_ = lambda;
  precision_ = precision;
  max_iterations_ = kMaxIterations;
  twist_init_ = kTwistInit;
  twist_ = twist_init_;
}

// TODO: check self-defined type initializer: Vector6d{0,0,0,0,0,0}
Vector6d LevenbergMarquardtOptimizer::Solve(const ImagePyramid& kImagePyr1,
                                            const DepthPyramid& kDepthPyr1,
                                            const ImagePyramid& kImagePyr2){
  OptimizerStatus status;
  status = OptimizeCameraPose(kImagePyr1, kDepthPyr1, kImagePyr2, twist_);
  if (status == -1) {
    std::cout << "Optimize failed! " << std::endl;
    return Vector6d{0,0,0,0,0,0};
  }
  else{
    return twist_;
  }
}

// The function that actually solves the optimization
OptimizerStatus LevenbergMarquardtOptimizer::OptimizeCameraPose(const ImagePyramid& kImagePyr1,
                                   const DepthPyramid& kDepthPyr1,
                                   const ImagePyramid& kImagePyr2,
                                   Vector6d& twist){
  return 0;
}

} // namespace odometry

