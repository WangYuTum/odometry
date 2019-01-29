// The file runs full pipline of odometry on live stereo camera.
// Camera parameters are read from calibration file.
// Multi-thread is used to guarantee real-time.
// Created by Yu Wang on 2019-01-13.

// Note:
// Camera Output:
//  * MUST be created with dynamic allocator (shared pointer)
//  * MUST be aligned to 32-bit address

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <fstream>
#include "include/camera.h"
#include "data_types.h"
#include "include/depth_estimate.h"
#include "include/image_processing_global.h"
#include "include/image_pyramid.h"
#include "include/lm_optimizer.h"
#include <se3.hpp>
#include <typeinfo>
#include <string>
#include "include/io_camera.h"
#include <boost/thread.hpp>

int main(){

  /***************************************** System initialisation ********************************************/
  std::cout << "Initializing odometry system ... Please KEEP CAMERA STILL ..." << std::endl;
  const int pyramid_levels = 4;
  std::shared_ptr<odometry::CameraPyramid> cam_ptr_left=nullptr, cam_ptr_right=nullptr;
  float optimizer_precision = 0.995f;

  // current frame pair. Producer: camera, Consumer: depth/pose/keyframe ...; TODO: corresponding lock, conditional var/notify_control
  cv::Mat current_left(480, 640, PixelType);
  cv::Mat current_right(480, 640, PixelType);




  /**************************************** Init Camera ********************************************/
  // create/setup stereo camera instance: call SetUpStereoCameraSystem()
  //  * this will create left/right camera pyramid with rectified intrinsics
  //  * this will also create valid regions, which will be needed later
  //  * intersect valid region with pre-defined valid boundary, multiple of 4
  const std::string calib_file = "../calibration_file/camchain.yaml";
  cv::Rect valid_region_rectify; // cv::Rect(top_left_x, top_left_y, width, height);
  double baseline;
  odometry::GlobalStatus cam_set_status = SetUpStereoCameraSystem(calib_file, pyramid_levels, cam_ptr_left, cam_ptr_right, valid_region_rectify, baseline);
  if (cam_set_status == -1){
    std::cout << "Init stereo cam system failed!" << std::endl;
    return -1;
  } else {
    std::cout << "Valid image region: " << valid_region_rectify << std::endl;
  }



  /**************************************** Init Depth Estimator ********************************************/
  float search_min = 0.1f; // in meters
  float search_max = 10.0f; // in meters
  int max_residuals = 10000; // max num of residuals per image
  float disparity_grad_th = 7.0f;
  float disparity_ssd_th = 1000.0f;
  float depth_photo_th = 5.0f;
  float depth_lambda = 0.01f;
  float depth_huber_delta = 28.0f;
  int depth_max_iters = 50;
  odometry::DepthEstimator depth_estimator(disparity_grad_th, disparity_ssd_th, depth_photo_th, search_min, search_max,
                                           depth_lambda, depth_huber_delta, optimizer_precision, depth_max_iters, valid_region_rectify,
                                           cam_ptr_left, cam_ptr_right, float(baseline), max_residuals);
  std::cout << "Created depth estimator." << std::endl;



  /**************************************** Init Pose Estimator ********************************************/
  std::vector<int> pose_max_iters = {10, 20, 30, 30}; // max_iters allowed for different pyramid levels
  odometry::Affine4f init_relative_affine;  // init relative pose, set to Identity by default
  init_relative_affine.block<3,3>(0,0) = Eigen::Matrix<float, 3, 3>::Identity();
  init_relative_affine.block<1,4>(3,0) << 0.0f, 0.0f, 0.0f, 1.0f;
  init_relative_affine.block<3,1>(0,3) << 0.0f, 0.0f, 0.0f;
  odometry::Affine4f cur_pose;
  cur_pose.block<3,3>(0,0) = Eigen::Matrix<float, 3, 3>::Identity();
  cur_pose.block<1,4>(3,0) << 0.0f, 0.0f, 0.0f, 1.0f;
  cur_pose.block<3,1>(0,3) << 0.0f, 0.0f, 0.0f;
  int robust_estimator = 1; // robust estimator: 0-no, 1-huber, 2-t_dist;
  float pose_huber_delta = 28.0f; // 4/255
  odometry::LevenbergMarquardtOptimizer pose_estimator(0.01f, optimizer_precision, pose_max_iters, init_relative_affine, cam_ptr_left, robust_estimator, pose_huber_delta);
  std::cout << "Created pose estimator." << std::endl;

  /**************************************** Cam I/O Thread ********************************************/
  // odometry::RunCamera(current_left, current_right, cam_ptr_left, cam_ptr_right, 0);
  std::cout << "Starting Camera I/O thread ..." << std::endl;
  cv::namedWindow("Left_rectified", cv::WINDOW_NORMAL);
  cv::namedWindow("Right_rectified", cv::WINDOW_NORMAL);
  boost::thread camera_io_thread(odometry::RunCamera, &current_left, &current_right, cam_ptr_left, cam_ptr_right, 0);
  std::cout << "Camera I/O thread started." << std::endl;

  /*
  cv::imshow("Left_rectified", current_left);
  cv::imshow("Right_rectified", current_right);
  cv::waitKey(5);
  */







  // create camera output buffer

  // create GUI (nanogui, and all other necessary windows)

  /********************************* Tracking ************************************/

  // Compute depth (need valid region)
  //  * output valid map, which will be used by tracking (do not need valid region anymore)

  // Compute pose (need valid map)

  camera_io_thread.join();

  return 0;
}