// The file runs full pipline of odometry on kitti stereo sequences.
// No real camera is used, camera parameters are hard-coded.
// No multi-thread used, only sequential pipeline
// Created by Yu Wang on 2019-01-13.

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <fstream>
#include "include/camera.h"
#include "data_types.h"
#include "include/depth_estimate.h"
#include "include/image_processing_global.h"
#include "include/image_pyramid.h"
#include "include/lm_optimizer.h"
#include <se3.hpp>
#include <typeinfo>

void load_data(const std::string& folder_name, std::vector<cv::Mat> &gray, Eigen::Matrix<float, 3, 4>& pose, int frame_id);
void eval_pose(const std::vector<Eigen::Matrix<float, 3, 4>>& gt_poses, const std::vector<Eigen::Matrix<float, 3, 4>> pred_poses);
void plot(const std::vector<Eigen::Matrix<float, 3, 4>>& gt_poses, const std::vector<Eigen::Matrix<float, 3, 4>> pred_poses);

int main(){

  // TODO: hardcode camera params in depth_estimate, lm_optimizer, WarpPixel, ReprojectToCameraFrame
  // Kitti sequence00, calibration
  unsigned int num_frames = 200;
  unsigned int num_pyramid = 4;
  std::string data_path = "../kitti";
  float fx = 718.856f; // in pixels
  float baseline = 386.1448f / 718.856f; // in meters: 0,53716572
  cv::Scalar init_val(0);
  std::vector<cv::Mat> pre_gray(2); // load for current frame's stereo img
  std::vector<cv::Mat> cur_gray(2); // load for current frame's stereo img
  Eigen::Matrix<float, 3, 4> gt_pose; // load for current frame's pose (left camera)
  std::vector<Eigen::Matrix<float, 3, 4>> gt_poses(num_frames); // store pose trajectory
  std::vector<Eigen::Matrix<float, 3, 4>> pred_poses(num_frames); // store pose trajectory


  std::cout << "Initializing odometry system ..." << std::endl;
  // initialise stereo cameras (null pointer since we only evaluate on kitti dataset)
  std::shared_ptr<odometry::CameraPyramid> left_cam_ptr = nullptr;
  std::shared_ptr<odometry::CameraPyramid> right_cam_ptr = nullptr;
  std::cout << "Created camera instance." << std::endl;


  // initialise depth estimator
  odometry::GlobalStatus depth_state;
  float search_min = 0.5f; // in meters
  float search_max = 20.0f; // in meters
  int max_residuals = 5000; // max num of residuals per image
  float disparity_grad_th = 35.0f;
  float disparity_ssd_th = 1000.0f;
  float depth_photo_th = 10.0f;
  float depth_lambda = 0.01f;
  float depth_huber_delta = 28.0f;
  float depth_precision = 0.995f;
  int depth_max_iters = 50;
  odometry::DepthEstimator depth_estimator(disparity_grad_th, disparity_ssd_th, depth_photo_th, search_min, search_max,
                                     depth_lambda, depth_huber_delta, depth_precision, depth_max_iters, num_pyramid,
                                     left_cam_ptr, right_cam_ptr, baseline, max_residuals);
  std::cout << "Created depth estimator." << std::endl;


  // initialise pose estimator
  std::vector<int> pose_max_iters = {10, 20, 30, 30}; // max_iters allowed for different pyramid levels
  odometry::Affine4f init_relative_affine;  // init relative pose, set to Identity by default
  init_relative_affine.block<3,3>(0,0) = Eigen::Matrix<float, 3, 3>::Identity();
  init_relative_affine.block<1,4>(3,0) << 0.0f, 0.0f, 0.0f, 1.0f;
  init_relative_affine.block<3,1>(0,3) << 0.0f, 0.0f, 0.0f;
  odometry::Affine4f rela_pose;
  odometry::Affine4f cur_pose;
  cur_pose.block<3,3>(0,0) = Eigen::Matrix<float, 3, 3>::Identity();
  cur_pose.block<1,4>(3,0) << 0.0f, 0.0f, 0.0f, 1.0f;
  cur_pose.block<3,1>(0,3) << 0.0f, 0.0f, 0.0f;
  int robust_estimator = 1; // robust estimator: 0-no, 1-huber, 2-t_dist;
  float pose_huber_delta = 28.0f;
  odometry::LevenbergMarquardtOptimizer pose_estimator(0.01f, 0.995f, pose_max_iters, init_relative_affine, left_cam_ptr, robust_estimator, pose_huber_delta);
  std::cout << "Created pose estimator." << std::endl;

  // initialise 0-th frame: compute left_depth
  load_data(data_path, pre_gray, gt_pose, 0);
  gt_poses.emplace_back(gt_pose);
  pred_poses.emplace_back(gt_pose);
  cv::Mat pre_left_val(pre_gray[0].rows, pre_gray[0].cols, CV_8U, init_val);
  cv::Mat pre_left_disp(pre_gray[0].rows, pre_gray[0].cols, PixelType, init_val);
  cv::Mat pre_left_dep(pre_gray[0].rows, pre_gray[0].cols, PixelType, init_val);
  depth_state = depth_estimator.ComputeDepth(pre_gray[0], pre_gray[1], pre_left_val, pre_left_disp, pre_left_dep);
  if (depth_state == -1){
    std::cout << "Init 0-th frame failed!" << std::endl;
    exit(-1);
  }
  odometry::ImagePyramid pre_img_pyramid(4, pre_gray[0], false);
  odometry::DepthPyramid pre_dep_pyramid(4, pre_left_disp, false);
  std::cout << "Initialize done." << std::endl << std::endl;


  // estimate pose from 1-th frame
  for (unsigned int frame_id = 1; frame_id < num_frames; frame_id++){
    // load data: gray-imgs, gt_poses(left camera)
    std::cout << "reading frame " << frame_id << " ..." << std::endl;
    load_data(data_path, cur_gray, gt_pose, frame_id);
    gt_poses.emplace_back(gt_pose);

    // create image-pyramid
    odometry::ImagePyramid cur_img_pyramid(4, cur_gray[0], false); // create pyramid for left image

    // estimate pose and store
    rela_pose = pose_estimator.Solve(pre_img_pyramid, pre_dep_pyramid, cur_img_pyramid);
    cur_pose = cur_pose * rela_pose.inverse();
    pred_poses.emplace_back(cur_pose.block<3,4>(0,0));

    // estimate depth & create depth-pyramid
    cv::Mat cur_left_val(cur_gray[0].rows, cur_gray[0].cols, CV_8U, init_val);
    cv::Mat cur_left_disp(cur_gray[0].rows, cur_gray[0].cols, PixelType);
    cv::Mat cur_left_dep(cur_gray[0].rows, cur_gray[0].cols, PixelType);
    depth_state = depth_estimator.ComputeDepth(cur_gray[0], cur_gray[1], cur_left_val, cur_left_disp, cur_left_dep);
    if (depth_state == -1){
      std::cout << "    depth failed!" << std::endl;
      break;
    } else {
      std::cout << "    compute depth done." << std::endl;
      std::cout << "    number of val depth: " << cv::sum(cur_left_val)[0] << std::endl;
      depth_estimator.ReportStatus();
    }
    odometry::DepthPyramid pre_dep_pyramid(4, cur_left_dep, false);
    odometry::ImagePyramid pre_img_pyramid(4, cur_gray[0], false);
    cv::waitKey(100);
  }
  std::cout << "sequence done!" << std::endl;
  eval_pose(gt_poses, pred_poses);

  return 0;
}

void load_data(const std::string& folder_name, std::vector<cv::Mat> &gray, Eigen::Matrix<float, 3, 4>& pose, int frame_id){

}

void eval_pose(const std::vector<Eigen::Matrix<float, 3, 4>>& gt_poses, const std::vector<Eigen::Matrix<float, 3, 4>> pred_poses){

}

void plot(const std::vector<Eigen::Matrix<float, 3, 4>>& gt_poses, const std::vector<Eigen::Matrix<float, 3, 4>> pred_poses){

}