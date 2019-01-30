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
void save_3d(const cv::Mat& valid_map, const cv::Mat& idepth_map, const odometry::Affine4f& abs_pose, std::shared_ptr<odometry::CameraPyramid>& cam_ptr_left, std::vector<std::vector<float>>& points_3d);
void write_file(const std::vector<std::vector<float>>& points_3d);

int main(){

  /***************************************** System initialisation ********************************************/
  std::cout << "Initializing odometry system ..." << std::endl;
  const int pyramid_levels = 4;
  std::shared_ptr<odometry::CameraPyramid> cam_ptr_left=nullptr, cam_ptr_right=nullptr;
  float optimizer_precision = 0.995f;
  cv::Scalar init_val(0);
  clock_t begin, end;

  // previous/current frame pair
  cv::Mat previous_left(480, 640, PixelType);
  cv::Mat previous_right(480, 640, PixelType);
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
  cv::Mat raw_in;
  cv::VideoCapture cam_cap;
  int cam_deviceID = 0;
  cam_cap.open(cam_deviceID);
  if (!cam_cap.isOpened()) {
    std::cout << "ERROR! Unable to open camera." << std::endl;
  } else {
    cam_cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280); // 1280 x 1.5 = 1920 , [720, 960]
    cam_cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480); // 480 x 1.5 = 720
  }



  /**************************************** Init Depth Estimator ********************************************/
  float search_min = 0.1f; // in meters
  float search_max = 5.0f; // in meters
  int max_residuals = 10000; // max num of residuals per image
  float disparity_grad_th = 3.0f;
  float disparity_ssd_th = 1500.0f;
  float depth_photo_th = 7.0f;
  float depth_lambda = 0.01f;
  float depth_huber_delta = 28.0f;
  int depth_max_iters = 50;
  odometry::GlobalStatus depth_state;
  cv::Mat pre_left_val(previous_left.rows, previous_left.cols, CV_8U, init_val);
  cv::Mat pre_left_disp(previous_left.rows, previous_left.cols, PixelType, init_val);
  cv::Mat pre_left_dep(previous_left.rows, previous_left.cols, PixelType, init_val);
  odometry::DepthEstimator depth_estimator(disparity_grad_th, disparity_ssd_th, depth_photo_th, search_min, search_max,
                                           depth_lambda, depth_huber_delta, optimizer_precision, depth_max_iters, valid_region_rectify,
                                           cam_ptr_left, cam_ptr_right, float(baseline), max_residuals);
  std::cout << "Created depth estimator." << std::endl;



  /**************************************** Init Pose Estimator ********************************************/
  std::vector<odometry::Affine4f> poses;
  std::vector<int> pose_max_iters = {10, 20, 30, 30}; // max_iters allowed for different pyramid levels
  odometry::Affine4f init_relative_affine;  // init relative pose, set to Identity by default
  init_relative_affine.block<3,3>(0,0) = Eigen::Matrix<float, 3, 3>::Identity();
  init_relative_affine.block<1,4>(3,0) << 0.0f, 0.0f, 0.0f, 1.0f;
  init_relative_affine.block<3,1>(0,3) << 0.0f, 0.0f, 0.0f;
  odometry::Affine4f cur_pose;
  odometry::Affine4f rela_pose;
  cur_pose.block<3,3>(0,0) = Eigen::Matrix<float, 3, 3>::Identity();
  cur_pose.block<1,4>(3,0) << 0.0f, 0.0f, 0.0f, 1.0f;
  cur_pose.block<3,1>(0,3) << 0.0f, 0.0f, 0.0f;
  int robust_estimator = 1; // robust estimator: 0-no, 1-huber, 2-t_dist;
  float pose_huber_delta = 28.0f; // 4/255
  odometry::LevenbergMarquardtOptimizer pose_estimator(0.01f, optimizer_precision, pose_max_iters, init_relative_affine, cam_ptr_left, robust_estimator, pose_huber_delta);
  std::cout << "Created pose estimator." << std::endl;




  std::vector<std::vector<float>> points_3d;
  // Main loop
  // ----------------------------------------------------------------------------------------
  // ----------------------------------------------------------------------------------------
  // ----------------------------------------------------------------------------------------
  // Get the 0-th frame
  cv::Mat key_points;
  cv::namedWindow("keypoints", cv::WINDOW_NORMAL);
  int count = 0;
  std::cout << "Stand by, KEEP CAMERA STILL ..." << std::endl;
  cv::waitKey(3000); // wait for 3s
  cam_cap.read(raw_in);
  cvtColor(raw_in, raw_in, cv::COLOR_RGB2GRAY); // to gray-scale
  raw_in.convertTo(raw_in, PixelType);  // to FP32
  cv::Mat left_img(raw_in, cv::Rect(0, 0, 640, 480));
  cv::Mat right_img(raw_in, cv::Rect(640, 0, 640, 480));
  cam_ptr_left->UndistortRectify(left_img, previous_left);
  cam_ptr_right->UndistortRectify(right_img, previous_right);
  if (raw_in.empty()){
    std::cout << "Read 0-th frame failed!" << std::endl;
    return -1;
  }
  // compute depth on 0-th frame
  depth_state = depth_estimator.ComputeDepth(previous_left, previous_right, pre_left_val, pre_left_disp, pre_left_dep);
  if (depth_state == -1) {
    std::cout << "Init 0-th frame failed!" << std::endl;
    return -1;
  }
  odometry::ImagePyramid pre_img_pyramid(pyramid_levels, previous_left, false);
  odometry::DepthPyramid pre_dep_pyramid(pyramid_levels, pre_left_dep, false);
  // save 0th pose as identity
  poses.push_back(init_relative_affine);
  std::cout << "Initialize 0-th frame done." << std::endl << std::endl;
  // show 0th frame keypoints
  previous_left.convertTo(key_points, cv::IMREAD_GRAYSCALE);
  for (int y=0; y<pre_left_val.rows; y++){
    for (int x=0; x<pre_left_val.cols; x++){
      if (pre_left_val.at<uint8_t>(y,x)==1){
        cv::circle(key_points, cv::Point(x,y), 4, cv::Scalar(0));
      }
    }
  }
  cv::imshow("keypoints", key_points);
  cv::waitKey(1);
  save_3d(pre_left_val, pre_left_dep, poses[0], cam_ptr_left, points_3d);
  count++;

  while (true){
    // read from camera
    begin = clock();
    cam_cap.read(raw_in);
    cvtColor(raw_in, raw_in, cv::COLOR_BGR2GRAY); // to gray-scale
    raw_in.convertTo(raw_in, PixelType);  // to FP32
    cv::Mat left_img_raw(raw_in, cv::Rect(0, 0, 640, 480)); // no mat copy, only create new header
    cv::Mat right_img_raw(raw_in, cv::Rect(640, 0, 640, 480)); // no mat copy, only create new header
    cam_ptr_left->UndistortRectify(left_img_raw, current_left);
    cam_ptr_right->UndistortRectify(right_img_raw, current_right);
    if (raw_in.empty()){
      std::cout << "Read frame failed!" << std::endl;
      break;
    }
    // build current image pyramid
    odometry::ImagePyramid cur_img_pyramid(4, current_left, false);

    // tracking
    rela_pose = pose_estimator.Solve(pre_img_pyramid, pre_dep_pyramid, cur_img_pyramid);
    pose_estimator.Reset(init_relative_affine, 0.01f);
    cur_pose = cur_pose * rela_pose.inverse();
    poses.push_back(cur_pose);

    // compute depth on current image pair, save to pre_left_val and pre_left_dep for next tracking
    pre_left_val = cv::Mat::zeros(pre_left_val.rows, pre_left_val.cols, CV_8U); // clear the valid map
    pre_left_disp = cv::Mat::zeros(pre_left_val.rows, pre_left_val.cols, PixelType); // clear the valid map
    pre_left_dep = cv::Mat::zeros(pre_left_val.rows, pre_left_val.cols, PixelType); // clear the valid map
    depth_state = depth_estimator.ComputeDepth(current_left, current_right, pre_left_val, pre_left_disp, pre_left_dep);
    if (depth_state == -1){
      std::cout << "Depth failed!" << std::endl;
      break;
    }
    end = clock();
    std::cout << "tracking frame " << count << ": " << double(end - begin) / CLOCKS_PER_SEC * 1000.0f << " ms." << std::endl;
    // build pyramids for next tracking
    odometry::DepthPyramid pre_dep_pyramid(pyramid_levels, pre_left_dep, false);
    odometry::ImagePyramid pre_img_pyramid(pyramid_levels, current_left, false);
    // show n-th frame keypoints
    current_left.convertTo(key_points, cv::IMREAD_GRAYSCALE);
    for (int y=0; y<pre_left_val.rows; y++){
      for (int x=0; x<pre_left_val.cols; x++){
        if (pre_left_val.at<uint8_t>(y,x)==1){
          cv::circle(key_points, cv::Point(x,y), 4, cv::Scalar(0));
        }
      }
    }
    cv::imshow("keypoints", key_points);
    cv::waitKey(1);
    // save 3d point cloud
    save_3d(pre_left_val, pre_left_dep, poses[count], cam_ptr_left, points_3d);
    count++;
    if (count == 499)
      break;
  }
  // write to .off file for meshlab visualize
  write_file(points_3d);

  // create camera output buffer
  // create GUI (nanogui, and all other necessary windows)

  /********************************* Tracking ************************************/
  // Compute depth (need valid region)
  //  * output valid map, which will be used by tracking (do not need valid region anymore)
  // Compute pose (need valid map)

  return 0;
}

void save_3d(const cv::Mat& valid_map, const cv::Mat& idepth_map, const odometry::Affine4f& abs_pose, std::shared_ptr<odometry::CameraPyramid>& cam_ptr_left, std::vector<std::vector<float>>& points_3d){
  float x_3d, y_3d, z_3d;
  Eigen::Matrix<float, 4, 1> camera_3d, world_3d;
  for (int y = 0; y < valid_map.rows; y++){
    for (int x = 0; x < valid_map.cols; x++){
      if (valid_map.at<uint8_t>(y, x) == 1){
        // 3d coord in camera view
        z_3d = 1.0f / idepth_map.at<float>(y,x);
        x_3d = z_3d * (x - cam_ptr_left->cx_float(0)) / cam_ptr_left->fx_float(0);
        y_3d = z_3d * (y - cam_ptr_left->cy_float(0)) / cam_ptr_left->fy_float(0);
        // to world view
        camera_3d << x_3d, y_3d, z_3d, 1.0f;
        world_3d = abs_pose * camera_3d;
        points_3d.emplace_back(std::vector<float>{world_3d(0), world_3d(1), world_3d(2)});
      } else
        continue;
    }
  }
}

void write_file(const std::vector<std::vector<float>>& points_3d){
  std::string save_file = "../data/point_cloud.off";
  std::ofstream pc_file;
  unsigned long num_points;

  pc_file.open(save_file, std::ios::out | std::ios::trunc);
  if (!pc_file.is_open()){
    std::cout << "open point cloud write file failed: " << save_file << std::endl;
    exit(-1);
  }
  pc_file << "OFF" << std::endl;
  // vertex_count face_count edge_count
  num_points = points_3d.size();
  pc_file << std::to_string(num_points) << " " << std::to_string(0) << " " << std::to_string(0) << std::endl;
  // x y z r g b a
  for (unsigned long id = 0; id < num_points; id++){
    pc_file << std::to_string(points_3d[id][0]) << " " << std::to_string(points_3d[id][1]) << " " << std::to_string(points_3d[id][2])
    << " " << std::to_string(125) << " " << std::to_string(125) << " " << std::to_string(125) << " " << std::to_string(0) << std::endl;
  }
  pc_file.close();
}