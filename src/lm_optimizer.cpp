// Created by Yu Wang on 03.12.18.
// Implementation of LM Optimizer class.

#include <lm_optimizer.h>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <image_processing_global.h>
#include <se3.hpp>
#include <camera.h>
#include <algorithm>
#include <opencv2/highgui.hpp>
#include <math.h>

namespace odometry
{

LevenbergMarquardtOptimizer::LevenbergMarquardtOptimizer(float lambda,
                                                         float precision,
                                                         const std::vector<int> kMaxIterations,
                                                         const Affine4f& kRelativeInit,
                                                         const std::shared_ptr<CameraPyramid>& kCameraPtr,
                                                         const int robust_est,
                                                         const float huber_delta){
  lambda_ = lambda;
  precision_ = precision;
  max_iterations_ = kMaxIterations;
  affine_init_ = kRelativeInit;
  SetIdentityTransform(affine_);
  for (int i = 0; i < 4; i++){
    iters_stat_.push_back(0);
    cost_stat_.push_back(std::vector<float>{0.0, 0.0});
  }
  if (kCameraPtr == NULL)
    std::cout << "LM Optimizer failed! Invalid camera pointer!" << std::endl;
  else
    camera_ptr_ = kCameraPtr;
  robust_est_ = robust_est;
  huber_delta_ = huber_delta;
}

LevenbergMarquardtOptimizer::~LevenbergMarquardtOptimizer(){
  // release ownership of camera pointer
  camera_ptr_.reset();
}

void LevenbergMarquardtOptimizer::SetIdentityTransform(Affine4f& in_mat){
  in_mat.block<3, 3>(0, 0) = Eigen::Matrix<float, 3, 3>::Identity();
  in_mat.block<1, 3>(3, 0) << 0.0, 0.0, 0.0;
  in_mat.block<4, 1>(0, 3) << 0.0, 0.0, 0.0, 0.0;
}

Affine4f LevenbergMarquardtOptimizer::Solve(const ImagePyramid& kImagePyr1,
                                            const DepthPyramid& kDepthPyr1,
                                            const ImagePyramid& kImagePyr2){
  OptimizerStatus status;
  status = OptimizeCameraPose(kImagePyr1, kDepthPyr1, kImagePyr2);
  //status = OptimizeCameraPoseSse(kImagePyr1, kDepthPyr1, kImagePyr2);
  if (status == -1) {
    std::cout << "Optimize failed! " << std::endl;
    Affine4f tmp;
    SetIdentityTransform(tmp);
    return tmp;
  }
  else{
    return affine_;
  }
}


// The function that actually solves the optimization
OptimizerStatus LevenbergMarquardtOptimizer::OptimizeCameraPose(const ImagePyramid& kImagePyr1,
                                   const DepthPyramid& kDepthPyr1,
                                   const ImagePyramid& kImagePyr2){
  Sophus::SE3<float> current_estimate(affine_init_); // current pose estimate, always the best current pose
  Sophus::SE3<float> inc_estimate(affine_init_); // tempted pose estimate used to update the current_estimate
  Sophus::SE3<float> last_estimate = current_estimate; // used to save previous best pose
  Sophus::SE3<float> delta; // the incremented pose
  int pyr_levels = kImagePyr1.GetNumberLevels();
  int l = pyr_levels-1;
  Eigen::Matrix<float, 6, Eigen::Dynamic> jtw;
  Eigen::Matrix<float, 6, 6> jtwj;
  Eigen::Matrix<float, 6, 6> linear_a;
  Eigen::Matrix<float, 6, 1> linear_b;
  Eigen::Matrix<float, Eigen::Dynamic, 6> jaco;
  Eigen::DiagonalMatrix<float, Eigen::Dynamic> weights;
  Eigen::Matrix<float, Eigen::Dynamic, 1> residuals;
  int num_residuals = 0;
  float current_lambda = 0.0f;
  // loop for each pyramid level
  while (l >= 0){
    // get respective images/depth map from current pyramid level as const reference
    const cv::Mat& kImg1 = kImagePyr1.GetPyramidImage(l); // CV_32F
    const cv::Mat& kImg2 = kImagePyr2.GetPyramidImage(l); // CV_32F
    const cv::Mat& kDep1 = kDepthPyr1.GetPyramidDepth(l); // CV_32F
    // check data types and matrix size
    if ((kImg1.rows != kImg2.rows) || (kImg1.rows != kDep1.rows)){
      std::cout << "Image rows don't match in LevenbergMarquardtOptimizer::OptimizeCameraPose()." << std::endl;
      return -1;
    }
    if ((kImg1.cols != kImg2.cols) || (kImg1.cols != kDep1.cols)){
      std::cout << "Image cols don't match in LevenbergMarquardtOptimizer::OptimizeCameraPose()." << std::endl;
      return -1;
    }
    if ((kImg1.type() != PixelType) || (kImg2.type() != PixelType) || (kDep1.type() != PixelType)){
      std::cout << "Image types don't match in LevenbergMarquardtOptimizer::OptimizeCameraPose()." << std::endl;
      return -1;
    }
    int iter_count = 0;
    float err_last = 1e+10;
    float err_now = 0.0;
    current_lambda = lambda_;
    float err_diff = 1e+10;
    inc_estimate = current_estimate;
    // initial increment twist, default constructed as identity. re-define for each pyramid
    while (max_iterations_[l]> iter_count){
      // std::cout << "level: " << l << ", iter: " << iter_count << std::endl;
      num_residuals = 0;
      //clock_t begin = clock();
      OptimizerStatus compute_status = ComputeResidualJacobianNaive(kImg1, kImg2, kDep1, inc_estimate.matrix(), jaco, weights, residuals, num_residuals, l);
      //clock_t end = clock();
      if (compute_status == -1){
        std::cout << "Evaluate Residual & Jacobian failed " << std::endl;
        return -1;
      }
      //std::cout << "eval res/jaco: " << double(end - begin) / CLOCKS_PER_SEC * 1000.0f << " ms" << std::endl;
      // compute jacobian succeed, proceed
      err_now = (float(1.0) / float(num_residuals)) * residuals.transpose() * weights * residuals;
      //std::cout << "pose err: " << err_now << std::endl;
      if (err_now > err_last){ // bad pose estimate, do not update pose
        // std::cout << "bad pose" << std::endl;
        current_lambda = current_lambda * 5.0f;
        if (current_lambda > 1e+5) { break; }
        current_estimate = last_estimate;
      } else{ // good pose estimate -> update pose, save previous pose
        current_estimate = inc_estimate;
        last_estimate = current_estimate;
        err_diff = err_now / err_last;
        if (err_diff > precision_) { break; }
        err_last = err_now;
        current_lambda = std::max(current_lambda / 5.0f, float(1e-5));
      }
      // solve the system
      jtw = jaco.transpose() * weights;
      jtwj = jtw * jaco;
      Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero(); // zero 6x6 matrix
      H.diagonal() = jtwj.diagonal();
      linear_b = - jtw * residuals;
      linear_a = jtwj + current_lambda * H;
      Vector6f delta_vec = linear_a.colPivHouseholderQr().solve(linear_b);
      delta = Sophus::SE3<float>::exp(delta_vec);
      inc_estimate = Sophus::SE3<float>(delta.matrix() * current_estimate.matrix());
      iter_count++;
    } // end optimize criteria loop
    l--;
  } // end pyramid loop
  affine_ = current_estimate.matrix(); // assign the optimised pose
  return 0;
}

// Naive impl, big loop over all pixels
OptimizerStatus LevenbergMarquardtOptimizer::ComputeResidualJacobianNaive(const cv::Mat& kImg1,
                                                                     const cv::Mat& kImg2,
                                                                     const cv::Mat& kDep1,
                                                                     const Affine4f& kTransform,
                                                                     Eigen::Matrix<float, Eigen::Dynamic, 6>& jaco,
                                                                     Eigen::DiagonalMatrix<float, Eigen::Dynamic>& weight,
                                                                     Eigen::Matrix<float, Eigen::Dynamic, 1>& residual,
                                                                     int& num_residual,
                                                                     int level){
  // declare local vars
  int kRows = kImg1.rows;
  int kCols = kImg1.cols;
  float grad_x, grad_y, mag_grad;
  int num_invalid_dep = 0;
  int num_out_bound = 0;
  float scale = 0.0f;
  float scale_sqr = 0.0f;
  RowVector2f grad(0.0, 0.0);
  Vector4f left_coord, left_3d, right_3d, warped_coordf;
  Vector2i warped_coordi;
  GlobalStatus warp_flag;
  GlobalStatus grad_flag;
  Matrix2ff jw;
  float fx_z, fy_z, xx, yy, zz, xy;
  residual.resize(kRows*kCols, 1);
  jaco.resize(kRows*kCols, 6);
  // loop over all pixels
  for (int y = 4; y < kRows - 4; y++){ // ignore boundary by 4 pixels
    for (int x = 4; x < kCols - 4; x++){ // ignore boundary by 4 pixels
      // skip invalid depth
      if (std::fabs(kDep1.at<float>(y, x) - 0.0f) < 0.01f) {
        num_invalid_dep++;
        continue;
      } else{
        //std::cout << y << " " << x << std::endl;
        left_coord << x, y, 1.0f / kDep1.at<float>(y, x), 1.0f;
        //std::cout << "Sec0" << std::endl;
        ReprojectToCameraFrame(left_coord, camera_ptr_, left_3d, level);
        //std::cout << "Sec1" << std::endl;
        warp_flag = WarpPixel(left_3d, kTransform, kRows, kCols, camera_ptr_, warped_coordf, right_3d, level);
        //std::cout << "Sec2" << std::endl;
        if (warp_flag == -1) { // out of image boundary
          num_out_bound++;
          continue;
        }
        warped_coordi(0) = int(std::floor(warped_coordf(0)));
        warped_coordi(1) = int(std::floor(warped_coordf(1)));
        //std::cout << "Sec3" << std::endl;
//        std::cout << "left_coord: " << left_coord << std::endl;
//        std::cout << "left_3d: " << left_3d << std::endl;
//        std::cout << "warped_coordf: " << warped_coordf(1) << " " <<  warped_coordf(0) << std::endl;
//        std::cout << "warped_coordi: " << warped_coordi(1) << " " <<  warped_coordi(0) << std::endl;
        ComputePixelGradient(kImg2, kRows, kCols, warped_coordi(1), warped_coordi(0), grad); //BUG!!!
        //std::cout << "Sec4" << std::endl;
        residual.row(num_residual) << kImg2.at<float>(warped_coordi(1), warped_coordi(0)) - kImg1.at<float>(y, x);
        //std::cout << "Sec5" << std::endl;
        // compute partial jacobian with left_3d
        // TODO: only for debug now
        // fx_z = camera_ptr_->fx(level) / left_3d(2);
        // fy_z = camera_ptr_->fy(level) / left_3d(2);
        fx_z = (718.856f / std::pow(2.0f, level)) / left_3d(2);
        fy_z = (718.856f / std::pow(2.0f, level)) / left_3d(2);
        xy = left_3d(0) * left_3d(1);
        xx = left_3d(0) * left_3d(0);
        yy = left_3d(1) * left_3d(1);
        zz = left_3d(2) * left_3d(2);
        // TODO: only for debug now
        //jw << fx_z, 0.0, -fx_z * left_3d(0) / left_3d(2), -fx_z * xy / left_3d(2), camera_ptr_->fx(level) * (1.0 + xx / zz), -fx_z * left_3d(1),
        //        0.0, fy_z, -fy_z * left_3d(1) / left_3d(2), -camera_ptr_->fy(level) * (1.0 + yy / zz),  fy_z * xy / left_3d(2), fy_z * left_3d(0);
        jw << fx_z, 0.0, -fx_z * left_3d(0) / left_3d(2), -fx_z * xy / left_3d(2), (718.856f / std::pow(2.0f, level)) * (1.0 + xx / zz), -fx_z * left_3d(1),
                0.0, fy_z, -fy_z * left_3d(1) / left_3d(2), -(718.856f / std::pow(2.0f, level)) * (1.0 + yy / zz),  fy_z * xy / left_3d(2), fy_z * left_3d(0);
        jaco.row(num_residual) = grad * jw;
        //std::cout << "Sec6" << std::endl;
        num_residual++;
      }
    }
  }
  //std::cout << "num of valid depth in pose_opt: " << num_residual  << " over total: " << kRows*kCols << std::endl;
  //std::cout << "num out of bound: " << num_out_bound << std::endl;
  residual.conservativeResize(num_residual, 1);
  jaco.conservativeResize(num_residual, 6);
  if (num_residual == 0 || jaco.rows() != num_residual || residual.rows() != num_residual){
    std::cout << "Num residual: " <<  num_residual << std::endl;
    std::cout << "jaco rows: " << jaco.rows() << std::endl;
    return -1;
  }
  weight.setIdentity(num_residual);
  if (robust_est_ == 0){
    return 0;
  } else if (robust_est_ == 1){
    for (int i = 0; i < num_residual; i++){
      weight.diagonal()(i) = std::fabs(residual(i)) <= huber_delta_ ? 1.0f : huber_delta_ / std::fabs(residual(i));
    }
  } else{
    scale = ComputeScaleNaive(residual, num_residual);
    scale_sqr = scale * scale;
    for (int i = 0; i < num_residual; i++){
      weight.diagonal()(i) = (200.0f + 1.0f) / (200.0f + residual(i) * residual(i) / scale_sqr);
    }
  }
  return 0;
}

OptimizerStatus LevenbergMarquardtOptimizer::OptimizeCameraPoseSse(const ImagePyramid& kImagePyr1,
                                                                   const DepthPyramid& kDepthPyr1,
                                                                   const ImagePyramid& kImagePyr2){
  Sophus::SE3<float> current_estimate(affine_init_); // current pose estimate, always the best current pose
  Sophus::SE3<float> inc_estimate(affine_init_); // tempted pose estimate used to update the current_estimate
  Sophus::SE3<float> last_estimate; // used to save previous best pose
  Sophus::SE3<float> delta; // the incremented pose
  int pyr_levels = kImagePyr1.GetNumberLevels();
  int l = pyr_levels-1;
  Eigen::Matrix<float, 6, 640*480> jtw; // the max possible size
  Eigen::Matrix<float, 6, 6> jtwj;
  Eigen::Matrix<float, 6, 6> linear_a;
  Eigen::Matrix<float, 6, 1> linear_b;
  Eigen::Matrix<float, 640*480, 6> jaco;
  Eigen::Matrix<float, 640*480, 1> weights;
  Eigen::Matrix<float, 640*480, 1> residuals;
  int num_residuals = 0;
  float current_lambda = 0.0f;
  // loop for each pyramid level
  while (l >= 0) {
    // get respective images/depth map from current pyramid level as const reference
    const cv::Mat& kImg1 = kImagePyr1.GetPyramidImage(l); // CV_32F
    const cv::Mat& kImg2 = kImagePyr2.GetPyramidImage(l); // CV_32F
    const cv::Mat& kDep1 = kDepthPyr1.GetPyramidDepth(l); // CV_32F
    // check data types and matrix size
    if ((kImg1.rows != kImg2.rows) || (kImg1.rows != kDep1.rows)){
      std::cout << "Image rows don't match in LevenbergMarquardtOptimizer::OptimizeCameraPose()." << std::endl;
      return -1;
    }
    if ((kImg1.cols != kImg2.cols) || (kImg1.cols != kDep1.cols)){
      std::cout << "Image cols don't match in LevenbergMarquardtOptimizer::OptimizeCameraPose()." << std::endl;
      return -1;
    }
    if ((kImg1.type() != PixelType) || (kImg2.type() != PixelType) || (kDep1.type() != PixelType)){
      std::cout << "Image types don't match in LevenbergMarquardtOptimizer::OptimizeCameraPose()." << std::endl;
      return -1;
    }
    int iter_count = 0;
    float err_last = 1e+10;
    float err_now = 0.0;
    current_lambda = lambda_;
    float err_diff = 1e+10;
    inc_estimate = current_estimate;
    // initial increment twist, default constructed as identity. re-define for each pyramid
    while (max_iterations_[l]> iter_count){
      std::cout << "level: " << l << ", iter: " << iter_count << std::endl;
      num_residuals = 0;
      clock_t begin = clock();
      // OptimizerStatus compute_status = ComputeResidualJacobianSse(kImg1, kImg2, kDep1, inc_estimate.matrix(), jaco, weights, residuals, num_residuals, l);
      clock_t end = clock();
//      if (compute_status == -1){
//        std::cout << "Evaluate Residual & Jacobian failed " << std::endl;
//        return -1;
//      }
      std::cout << "eval res/jaco: " << double(end - begin) / CLOCKS_PER_SEC * 1000.0f << " ms" << std::endl;
    }
  }
  return 0;
}

// TODO, SSE impl, highly optimized
OptimizerStatus LevenbergMarquardtOptimizer::ComputeResidualJacobianSse(const cv::Mat& kImg1, const cv::Mat& kImg2, const cv::Mat& kDep1, const Affine4f& kTransform,
                                           Eigen::Matrix<float, Eigen::Dynamic, 6>& jaco,
                                           Eigen::DiagonalMatrix<float, Eigen::Dynamic, Eigen::Dynamic>& weight,
                                           Eigen::VectorXf& residual,
                                           int& num_residual){
  int kRows = kImg1.rows;
  int kCols = kImg1.cols;
  return 0;
}


float LevenbergMarquardtOptimizer::ComputeScaleNaive(const Eigen::VectorXf& residual, const int num_residual){
  float init_sigma = 5.0f;
  float vee = 200.0f;
  float current_sigma = init_sigma;
  float sigma_sqr = 0.0f;
  float sum = 0.0f;
  float err_sqr = 0.0f;
  do {
    init_sigma = current_sigma;
    sigma_sqr = current_sigma * current_sigma;
    sum = 0.0f;
    // update current_sigma
    for (int i = 0; i < num_residual; i++){
      err_sqr = residual(i) * residual(i);
      sum +=  err_sqr * (1.0f + vee) / (vee + err_sqr / sigma_sqr);
    }
    current_sigma = std::sqrtf(sum / float(num_residual));
  }while (std::fabs(current_sigma - init_sigma) >= float(1e-3));

  return current_sigma;
}

// TODO, SSE impl, highly optimized
float LevenbergMarquardtOptimizer::ComputeScaleSse(const Eigen::VectorXf& residual, const int num_residual){
}

void LevenbergMarquardtOptimizer::ShowReport(){
  std::cout << "Number of iterations performed per level: ";
  std::cout << iters_stat_[0] << ", " << iters_stat_[1] << ", " << iters_stat_[2] << ", " << iters_stat_[3] << std::endl;
  std::cout << "Costs before/after per level: " << std::endl;
  for (int i = 0; i < 4; i++){
    std::cout << cost_stat_[i][0] << ", " << cost_stat_[i][1] << std::endl;
  }
}

OptimizerStatus LevenbergMarquardtOptimizer::Reset(const Affine4f& kRelativeInit, const float lambda){
  OptimizerStatus status_set_init = SetInitialAffine(kRelativeInit);
  OptimizerStatus status_set_lambda = SetLambda(lambda);
  OptimizerStatus status_set_stat = ResetStatistics();
  if (status_set_init == -1 || status_set_lambda == -1 || status_set_stat == -1){
    std::cout << "Reset optimizer failed!" << std::endl;
    return -1;
  }
  return 0;
}


OptimizerStatus LevenbergMarquardtOptimizer::SetInitialAffine(const Affine4f& kAffineInit){
  affine_init_ = kAffineInit;
  // return -1;
  return 0;
}

OptimizerStatus LevenbergMarquardtOptimizer::SetLambda(const float lambda = 0.001){
  lambda_ = lambda;
  // return -1;
  return 0;
}

OptimizerStatus LevenbergMarquardtOptimizer::ResetStatistics(){
  // set all statistics to zeros
  for (int i = 0; i < 4; i++){
    iters_stat_[i] = 0;
    cost_stat_[i] = {0.0, 0.0};
  }
  // return -1;
  return 0;
}

} // namespace odometry