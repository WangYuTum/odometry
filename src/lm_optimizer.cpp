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

namespace odometry
{
/*
LevenbergMarquardtOptimizer::LevenbergMarquardtOptimizer() {
  lambda_ = 0.001;
  precision_ = 5e-7;
  // set identity rigid group
  SetIdentityTransform(twist_init_);
  twist_ = twist_init_;
  for (int i = 0; i < 4; i++){
    max_iterations_.push_back(100);
    iters_stat_.push_back(0);
    cost_stat_.push_back(std::vector<float>{0.0, 0.0});
  }
  camera_ptr_ = NULL;
}
*/

LevenbergMarquardtOptimizer::LevenbergMarquardtOptimizer(float lambda,
                                                         float precision,
                                                         const std::vector<int> kMaxIterations,
                                                         const Matrix44f& kTwistInit,
                                                         const std::shared_ptr<Camera>& kCameraPtr){
  lambda_ = lambda;
  precision_ = precision;
  max_iterations_ = kMaxIterations;
  twist_init_ = kTwistInit;
  SetIdentityTransform(twist_);
  for (int i = 0; i < 4; i++){
    iters_stat_.push_back(0);
    cost_stat_.push_back(std::vector<float>{0.0, 0.0});
  }
  if (kCameraPtr == NULL){
    std::cout << "LM Optimizer failed! Invalid camera pointer!" << std::endl;
    // terminate programe
  } else
    camera_ptr_ = kCameraPtr;
}

LevenbergMarquardtOptimizer::~LevenbergMarquardtOptimizer(){
  // release ownership of camera pointer
  camera_ptr_.reset();
}

void LevenbergMarquardtOptimizer::SetIdentityTransform(Matrix44f& in_mat){
  in_mat.block<3, 3>(0, 0) = Eigen::Matrix<float, 3, 3>::Identity();
  in_mat.block<1, 3>(3, 0) << 0.0, 0.0, 0.0;
  in_mat.block<4, 1>(0, 3) << 0.0, 0.0, 0.0, 0.0;
}

Matrix44f LevenbergMarquardtOptimizer::Solve(const ImagePyramid& kImagePyr1,
                                            const DepthPyramid& kDepthPyr1,
                                            const ImagePyramid& kImagePyr2){
  OptimizerStatus status;
  status = OptimizeCameraPose(kImagePyr1, kDepthPyr1, kImagePyr2);
  if (status == -1) {
    std::cout << "Optimize failed! " << std::endl;
    Matrix44f tmp;
    SetIdentityTransform(tmp);
    return tmp;
  }
  else{
    return twist_;
  }
}


// The function that actually solves the optimization
OptimizerStatus LevenbergMarquardtOptimizer::OptimizeCameraPose(const ImagePyramid& kImagePyr1,
                                   const DepthPyramid& kDepthPyr1,
                                   const ImagePyramid& kImagePyr2){
  // initial increment twist, default constructed as identity
  Sophus::SE3<float> delta_twist;
  // updated twist only used internally in this function, assigned to input arg: twist after optimization
  Sophus::SE3<float> update_twist(twist_init_);
  int pyr_levels = kImagePyr1.GetNumberLevels();
  int l = pyr_levels-1;
  Eigen::Matrix<float, 6, Eigen::Dynamic> jtw;
  Eigen::Matrix<float, 6, 6> jtwj;
  Eigen::Matrix<float, 6, 6> linear_a;
  Eigen::Matrix<float, 6, 1> linear_b;
  Eigen::Matrix<float, Eigen::Dynamic, 6> jaco;
  Eigen::DiagonalMatrix<float, Eigen::Dynamic, Eigen::Dynamic> weights;
  Eigen::VectorXf residuals;
  int num_residuals = 0;
  // loop for each pyramid level
  while (l >= 3){
    // get respective images/depth map from current pyramid level as const reference
    const cv::Mat& kImg1 = kImagePyr1.GetPyramidImage(l); // CV_32F
    const cv::Mat& kImg2 = kImagePyr2.GetPyramidImage(l); // CV_32F
    const cv::Mat& kDep1 = kDepthPyr1.GetPyramidDepth(l); // CV_32F
    std::cout << "level: " << l << std::endl;
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
    float err_diff = err_last - err_now;
    while (err_diff > precision_ && max_iterations_[l]> iter_count){
      std::cout << "iter: " << iter_count << std::endl;
      num_residuals = 0;
      // TODO: check how to increment
      Matrix44f incre_trans = update_twist.matrix() * delta_twist.matrix();
      clock_t begin = clock();
      OptimizerStatus compute_status = ComputeResidualJacobianNaive(kImg1, kImg2, kDep1, incre_trans, jaco, weights, residuals, num_residuals);
      clock_t end = clock();
      if (compute_status == -1){
        std::cout << "Evaluate Residual & Jacobian failed " << std::endl;
        return -1;
      }
      std::cout << "res/jaco success: " << double(end - begin) / CLOCKS_PER_SEC * 1000.0f << " ms" << std::endl;
      // compute jacobian succeed, proceed
      begin = clock();
      err_now = (float(1.0) / float(num_residuals)) * residuals.transpose() * weights * residuals;
      end = clock();
      std::cout << "cur_err: " << err_now << " in: "<< double(end - begin) / CLOCKS_PER_SEC * 1000.0f << " ms" << std::endl;
      if (iter_count == 0)
        cost_stat_[l][0] = err_now;
      if (err_now > err_last){
        std::cout << "energy increased" << std::endl;
        delta_twist = Sophus::SE3<float>();
      } else{
        std::cout << "energy decreased" << std::endl;
        cost_stat_[l][1] = err_now;
        // udpate current pose and errors
        update_twist = Sophus::SE3<float>(incre_trans);
        err_diff = err_last - err_now;
        err_last = err_now;
        // lambda_ = std::max(lambda_ / float(10.0), float(1e-7));
        // solve linear system
        begin = clock();
        jtw = jaco.transpose() * weights;
        jtwj = jtw * jaco;
        Eigen::Matrix<float, 6, 6>  damp_mat = Eigen::Matrix<float, 6, 6>::Zero();
        damp_mat.diagonal() = jtwj.diagonal();
        linear_a = jtwj + lambda_ * damp_mat;
        linear_b = - jtw * residuals;
        Vector6f delta = linear_a.colPivHouseholderQr().solve(linear_b);
        end = clock();
        std::cout << "solve time: " << double(end - begin) / CLOCKS_PER_SEC * 1000.0f << " ms" << std::endl;
        delta_twist = delta_twist.exp(delta);

      }
      std::cout << "num res: " << num_residuals << std::endl;
      std::cout << "res: " << std::endl << residuals.block<10,1>(0,0) << std::endl;
      iter_count++;
    } // end optimize criteria loop
    iters_stat_[l] = iter_count;
    l--;
  } // end pyramid loop
  twist_ = update_twist.matrix(); // assign the optimised pose
  return 0;
}

// Naive impl, big loop over all pixels
// TODO: openmp
OptimizerStatus LevenbergMarquardtOptimizer::ComputeResidualJacobianNaive(const cv::Mat& kImg1,
                                                                     const cv::Mat& kImg2,
                                                                     const cv::Mat& kDep1,
                                                                     const Matrix44f& kTranform,
                                                                     Eigen::Matrix<float, Eigen::Dynamic, 6>& jaco,
                                                                     Eigen::DiagonalMatrix<float, Eigen::Dynamic, Eigen::Dynamic>& weight,
                                                                     Eigen::VectorXf& residual,
                                                                     int& num_residual){
  // declare local vars
  int kRows = kImg1.rows;
  int kCols = kImg1.cols;
  int num_invalid_dep = 0;
  int num_out_bound = 0;
  RowVector2f grad(0.0, 0.0);
  Vector4f left_coord, left_3d, warped_coordf;
  Vector2i warped_coordi;
  GlobalStatus warp_flag;
  Matrix2ff jw;
  float fx_z, fy_z, xx, yy, zz, xy;
  // loop over all pixels
  for (int y = 0; y < kRows; y++){
    for (int x = 0; x < kCols; x++){
      // skip invalid depth
      if (kDep1.at<float>(y, x) == 0){
        num_invalid_dep++;
        continue;
      } else{
        left_coord << x, y, kDep1.at<float>(y, x), 1;
        ReprojectToCameraFrame(left_coord, camera_ptr_, left_3d);
        warp_flag = WarpPixel(left_3d, kTranform, kRows, kCols, camera_ptr_, warped_coordf);
        if (warp_flag == -1) { // out of image boundary
          num_out_bound++;
          continue;
        }
        warped_coordi(0) = int(std::floor(warped_coordf(0)));
        warped_coordi(1) = int(std::floor(warped_coordf(1)));
        residual.conservativeResize(num_residual+1, 1);
        residual.row(num_residual) << kImg2.at<float>(warped_coordi(1), warped_coordi(0)) - kImg1.at<float>(y, x);
        // compute gradient on the warped coordinate
        ComputePixelGradient(kImg2, kRows, kCols, warped_coordi(1), warped_coordi(0), grad);
        // compute partial jacobian
        fx_z = camera_ptr_->fx_ / left_3d(2);
        fy_z = camera_ptr_->fy_ / left_3d(2);
        xy = left_3d(0) * left_3d(1);
        xx = left_3d(0) * left_3d(0);
        yy = left_3d(1) * left_3d(1);
        zz = left_3d(2) * left_3d(2);
        jw << fx_z, 0.0, -fx_z * left_3d(0) / left_3d(2), -fx_z * xy / left_3d(2), camera_ptr_->fx_ * (1.0 + xx / zz), -fx_z * left_3d(1),
                0.0, fy_z, -fy_z * left_3d(1) / left_3d(2), -camera_ptr_->fy_ * (1.0 + yy / zz),  fy_z * xy / left_3d(2), fy_z * left_3d(0);
        jaco.conservativeResize(num_residual+1, 6);
        jaco.row(num_residual) = grad * jw;
        num_residual++;
      }
    }
  }
  std::cout << "num invalid dep: " << num_invalid_dep << std::endl;
  std::cout << "num out bound: " << num_out_bound << std::endl;
  if (num_residual == 0 || jaco.rows() != num_residual || residual.rows() != num_residual){
    std::cout << "Num residual: " <<  num_residual << std::endl;
    std::cout << "jaco rows: " << jaco.rows() << std::endl;
    return -1;
  }
  weight.setIdentity(num_residual);
  return 0;
}

// TODO, SSE impl, highly optimized
OptimizerStatus LevenbergMarquardtOptimizer::ComputeResidualJacobianSse(const cv::Mat& kImg1, const cv::Mat& kImg2, const cv::Mat& kDep1, const Matrix44f& twist,
                                           Eigen::Matrix<float, Eigen::Dynamic, 6>& jaco,
                                           Eigen::DiagonalMatrix<float, Eigen::Dynamic, Eigen::Dynamic>& weight,
                                           Eigen::VectorXf& residual,
                                           int& num_residual){
  int kRows = kImg1.rows;
  int kCols = kImg1.cols;
  // cv::Mat warped_img(kRows, kCols, PixelType);
  // WarpImage(kImg2, kTranform, warped_img);
  // !!! check valid depth value and image boundary
  return 0;
}

void LevenbergMarquardtOptimizer::ShowReport(){
  std::cout << "Number of iterations performed per level: ";
  std::cout << iters_stat_[0] << ", " << iters_stat_[1] << ", " << iters_stat_[2] << ", " << iters_stat_[3] << std::endl;
  std::cout << "Costs before/after per level: " << std::endl;
  for (int i = 0; i < 4; i++){
    std::cout << cost_stat_[i][0] << ", " << cost_stat_[i][1] << std::endl;
  }
}

OptimizerStatus LevenbergMarquardtOptimizer::Reset(const Matrix44f& kTwistInit, const float lambda){
  OptimizerStatus status_set_init = SetInitialTwist(kTwistInit);
  OptimizerStatus status_set_lambda = SetLambda(lambda);
  OptimizerStatus status_set_stat = ResetStatistics();
  if (status_set_init == -1 || status_set_lambda == -1 || status_set_stat == -1){
    std::cout << "Reset optimizer failed!" << std::endl;
    return -1;
  }
  return 0;
}


OptimizerStatus LevenbergMarquardtOptimizer::SetInitialTwist(const Matrix44f& kTwistInit){
  twist_init_ = kTwistInit;
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

