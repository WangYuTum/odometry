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
#include <string>
#include <tuple>

void load_gt_pose(const std::string& folder_name, std::vector<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>& gt_poses);
void load_data(const std::string& folder_name, std::vector<cv::Mat> &gray, int frame_id);
void eval_pose(const std::vector<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>& gt_poses, const std::vector<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>& pred_poses);
void save_txt(const std::vector<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>& gt_poses, const std::vector<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>& pred_poses);
void save_to_vis(const std::vector<std::tuple<odometry::ImagePyramid, odometry::DepthPyramid, cv::Mat>>& data_vec, const std::vector<int>& keyframe_ids);

int main(){

  // TODO: hardcode camera params in depth_estimate, lm_optimizer, WarpPixel, ReprojectToCameraFrame
  // Kitti sequence00, calibration
  unsigned int num_frames = 130; // 4000
  unsigned int num_pyramid = 4;
  std::string data_path = "../dataset/kitti";
  float fx = 718.856f; // in pixels
  float cx = 607.1928; // in pixels
  float cy = 185.2157; // in pixels
  float baseline = 386.1448f / 718.856f; // in meters: 0,53716572
  cv::Scalar init_val(0);
  std::vector<cv::Mat> pre_gray(2); // load for previous frame's stereo img
  std::vector<cv::Mat> cur_gray(2); // load for current frame's stereo img
  std::vector<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> gt_poses(num_frames); // store gt pose trajectory
  std::vector<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> pred_poses(num_frames); // store pred pose trajectory


  std::cout << "Initializing odometry system ..." << std::endl;
  // initialise stereo cameras (null pointer since we only evaluate on kitti dataset)
  std::shared_ptr<odometry::CameraPyramid> left_cam_ptr = nullptr;
  std::shared_ptr<odometry::CameraPyramid> right_cam_ptr = nullptr;
  std::cout << "Created camera instance." << std::endl;


  // initialise depth estimator
  odometry::GlobalStatus depth_state;
  float search_min = 0.1f; // in meters
  float search_max = 30.0f; // in meters
  int max_residuals = 80000; // max num of residuals per image
  float disparity_grad_th = 8.0f;
  float disparity_ssd_th = 900.0f;
  float depth_photo_th = 15.0f;
  float depth_lambda = 0.01f;
  float depth_huber_delta = 28.0f;
  float depth_precision = 0.995f;
  int depth_max_iters = 50;
  odometry::DepthEstimator depth_estimator(disparity_grad_th, disparity_ssd_th, depth_photo_th, search_min, search_max,
                                     depth_lambda, depth_huber_delta, depth_precision, depth_max_iters, 4,
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
  odometry::Affine4f pose_to_keyframe;
  cur_pose.block<3,3>(0,0) = Eigen::Matrix<float, 3, 3>::Identity();
  cur_pose.block<1,4>(3,0) << 0.0f, 0.0f, 0.0f, 1.0f;
  cur_pose.block<3,1>(0,3) << 0.0f, 0.0f, 0.0f;
  int robust_estimator = 1; // robust estimator: 0-no, 1-huber, 2-t_dist;
  float pose_huber_delta = 28.0f; // 4/255
  odometry::LevenbergMarquardtOptimizer pose_estimator(0.01f, 0.995f, pose_max_iters, init_relative_affine, left_cam_ptr, robust_estimator, pose_huber_delta);
  std::cout << "Created pose estimator." << std::endl;

  // load gt poses
  load_gt_pose(data_path, gt_poses);

  // initialise 0-th frame: compute left_depth
  load_data(data_path, pre_gray, 0);
  pred_poses[0] = gt_poses[0];
  cur_pose.block<3,4>(0,0) = gt_poses[0];
  pose_to_keyframe = cur_pose;
  cv::Mat pre_left_val(pre_gray[0].rows, pre_gray[0].cols, CV_8U, init_val);
  cv::Mat pre_left_disp(pre_gray[0].rows, pre_gray[0].cols, PixelType, init_val);
  cv::Mat pre_left_dep(pre_gray[0].rows, pre_gray[0].cols, PixelType, init_val);
  depth_state = depth_estimator.ComputeDepth(pre_gray[0], pre_gray[1], pre_left_val, pre_left_disp, pre_left_dep);
  if (depth_state == -1){
    std::cout << "Init 0-th frame failed!" << std::endl;
    exit(-1);
  }
  depth_estimator.ReportStatus();
  cv::Mat gray_left;
//  pre_gray[0].convertTo(gray_left, cv::IMREAD_GRAYSCALE);
//  for (int y=0; y<pre_left_val.rows; y++){
//    for (int x=0; x<pre_left_val.cols; x++){
//      if (pre_left_val.at<uint8_t>(y,x)==1){
//        cv::circle(gray_left, cv::Point(x,y), 4, cv::Scalar(0));
//      }
//    }
//  }
//  cv::namedWindow("keypoints", cv::WINDOW_NORMAL);
//  cv::imshow("keypoints", gray_left);
//  cv::waitKey(1);
//  pre_left_dep.convertTo(gray_left, cv::IMREAD_GRAYSCALE, 125);
//  cv::namedWindow("valid_map", cv::WINDOW_NORMAL);
//  cv::imshow("valid_map", gray_left);
//  cv::waitKey(0);

  std::vector<odometry::ImagePyramid> img_pyr_vec;
  std::vector<odometry::DepthPyramid> img_dep_vec;

//  odometry::ImagePyramid pre_img_pyramid(4, pre_gray[0], true);
//  odometry::DepthPyramid pre_dep_pyramid(4, pre_left_dep, false);
  auto* pre_img_pyramid_ptr = new odometry::ImagePyramid(4, pre_gray[0], true);
  auto* pre_dep_pyramid_ptr = new odometry::DepthPyramid(4, pre_left_dep, false);
//  img_pyr_vec.emplace_back(odometry::ImagePyramid(4, pre_gray[0], true));
//  img_dep_vec.emplace_back(odometry::DepthPyramid(4, pre_left_dep, false));
  std::cout << "Initialize 0-th frame done." << std::endl << std::endl;

  // simulate keyframe
  std::vector<int> keyframe_id;
  std::vector<std::tuple<odometry::ImagePyramid, odometry::DepthPyramid, cv::Mat>> keyframes;
  std::vector<odometry::Affine4f> keyframe_poses_abs;
  unsigned int current_kf = 0;
  keyframes.emplace_back(std::make_tuple(*pre_img_pyramid_ptr, *pre_dep_pyramid_ptr, pre_left_val));
  keyframe_id.push_back(current_kf);
  keyframe_poses_abs.emplace_back(cur_pose);
  Eigen::Matrix<float, 6, 1> keyframe_weight;
  keyframe_weight << 0.1f/3.3f, 1.0f/3.3f, 0.1f/3.3f, 1.0f/3.3f, 0.1f/3.3f, 1.0f/3.3f;
  Eigen::Matrix<float, 1, 6> current_mot;
  std::cout << "****************************************** new keyframe:" << current_kf << " *********************"<< std::endl;

  int sum = 0;
  gray_left = pre_dep_pyramid_ptr->GetPyramidDepth(0);
  for (int y= 0; y < gray_left.rows; y++){
    for (int x = 0; x < gray_left.cols; x++){
      if (gray_left.at<float>(y,x) != 0)
        sum++;
    }
  }
  std::cout << "num of valid depth at level 0: " << sum << std::endl;
  gray_left = pre_dep_pyramid_ptr->GetPyramidDepth(1);
  sum = 0;
  for (int y= 0; y < gray_left.rows; y++){
    for (int x = 0; x < gray_left.cols; x++){
      if (gray_left.at<float>(y,x) != 0)
        sum++;
    }
  }
  std::cout << "num of valid depth at level 1: " << sum << std::endl;
  gray_left = pre_dep_pyramid_ptr->GetPyramidDepth(2);
  sum = 0;
  for (int y= 0; y < gray_left.rows; y++){
    for (int x = 0; x < gray_left.cols; x++){
      if (gray_left.at<float>(y,x) != 0)
        sum++;
    }
  }
  std::cout << "num of valid depth at level 2: " << sum << std::endl;
  gray_left = pre_dep_pyramid_ptr->GetPyramidDepth(3);
  sum = 0;
  for (int y= 0; y < gray_left.rows; y++){
    for (int x = 0; x < gray_left.cols; x++){
      if (gray_left.at<float>(y,x) != 0)
        sum++;
    }
  }
  std::cout << "num of valid depth at level 3: " << sum << std::endl;

//  pre_dep_pyramid.GetPyramidDepth(2).convertTo(gray_left, cv::IMREAD_GRAYSCALE, 255);
//  cv::imshow("level1", gray_left);
//  gray_left = pre_img_pyramid.GetPyramidImage(2);
//  cv::imshow("level2", gray_left);
//  gray_left = pre_img_pyramid.GetPyramidImage(3);
//  cv::imshow("level3", gray_left);
//  cv::waitKey(0);

//  return 0;


  // estimate pose from 1-th frame
  for (unsigned int frame_id = 1; frame_id < num_frames; frame_id++){
    // load data: gray-imgs, gt_poses(left camera)
    load_data(data_path, cur_gray, frame_id);
    std::cout << "read frame done " << std::endl;

    // create image-pyramid for current frame
    //img_pyr_vec.emplace_back(odometry::ImagePyramid(num_pyramid, cur_gray[0], true));
    odometry::ImagePyramid cur_img_pyramid(num_pyramid, cur_gray[0], true); // create pyramid for left image
//    cv::Mat tmp, show_tmp;
//    tmp = img_pyr_vec[frame_id].GetPyramidImage(3);
//    tmp.convertTo(show_tmp, cv::IMREAD_GRAYSCALE);
//    cv::imshow("show_tmp"+std::to_string(frame_id), show_tmp);
//    cv::waitKey(0);

    // estimate pose and store
    std::cout << "computing pose " << std::endl;
    // pose to current keyframe
    pose_to_keyframe = pose_estimator.Solve(std::get<0>(keyframes[current_kf]), std::get<1>(keyframes[current_kf]), cur_img_pyramid);
    std::cout << "compute pose done" << std::endl;
    // pose to world origin: concatenate with current keyframe abs pose
    cur_pose = keyframe_poses_abs[current_kf] * pose_to_keyframe.inverse();
    //pose_estimator.Reset(rela_pose, 0.01f);
    //pose_estimator.Reset(init_relative_affine, 0.01f);
    //cur_pose = rela_pose.inverse();
    pred_poses[frame_id] = cur_pose.block<3,4>(0,0);


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
//      cur_gray[0].convertTo(gray_left, cv::IMREAD_GRAYSCALE);
//      for (int y=0; y<cur_left_val.rows; y++){
//        for (int x=0; x<cur_left_val.cols; x++){
//          if (cur_left_val.at<uint8_t>(y,x)==1){
//            cv::circle(gray_left, cv::Point(x,y), 4, cv::Scalar(0));
//          }
//        }
//      }
//      cv::imshow("keypoints", gray_left);
//      cv::waitKey(0);
      //img_dep_vec.emplace_back(odometry::DepthPyramid(4, cur_left_dep, false));
    }
    delete pre_img_pyramid_ptr;
    delete pre_dep_pyramid_ptr;
    pre_img_pyramid_ptr = new odometry::ImagePyramid(num_pyramid, cur_gray[0], true);
    pre_dep_pyramid_ptr = new odometry::DepthPyramid(num_pyramid, cur_left_dep, false);
    // TODO: if pose to keyframe is larger than TH, add current image/depth as new keyframe, set init_pose as identity
    Sophus::SO3<float> rotation(pose_to_keyframe.block<3,3>(0,0));
    current_mot << std::fabs(rotation.angleX()), std::fabs(rotation.angleY()), std::fabs(rotation.angleZ()),
            std::fabs(pose_to_keyframe(0,3)), std::fabs(pose_to_keyframe(1,3)), std::fabs(pose_to_keyframe(2,3));
    float motion_mag = current_mot.dot(keyframe_weight);
    if ( motion_mag > 1.1f){
      keyframes.emplace_back(*pre_img_pyramid_ptr, *pre_dep_pyramid_ptr, cur_left_val);
      keyframe_poses_abs.emplace_back(cur_pose);
      pose_estimator.Reset(pose_to_keyframe, 0.01f);
      //pose_estimator.Reset(init_relative_affine, 0.01f);
      current_kf++;
      keyframe_id.push_back(frame_id);
      std::cout << "****************************************** new keyframe: " << current_kf << " *********************" << std::endl;
    } else {
      // TODO: else set init_pose as previous pose w.r.t the keyframe
      pose_estimator.Reset(pose_to_keyframe, 0.01f);
      std::cout << "****************************************** motion: " << motion_mag << " *********************" << std::endl;
    }
  }
  if (pre_img_pyramid_ptr != nullptr)
    delete pre_img_pyramid_ptr;
  if (pre_dep_pyramid_ptr != nullptr)
    delete pre_dep_pyramid_ptr;
  std::cout << "Sequence done! Evaluating translation error for the first 50 frames ..." << std::endl;
  eval_pose(gt_poses, pred_poses);
  std::cout << "Total keyframes: " << current_kf << std::endl;
  std::cout << "Saving poses for KITTI plot ..." << std::endl;
  save_txt(gt_poses, pred_poses);
  std::cout << "Saving data for visualize ..." << std::endl;
  save_to_vis(keyframes, keyframe_id);

  return 0;
}

void load_gt_pose(const std::string& folder_name, std::vector<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>& gt_poses){
  unsigned int num_frame = gt_poses.size();
  unsigned int scaner;
  std::string pose_txt = folder_name + "/poses/00.txt";
  std::ifstream pose_file;
  std::string pose_line;
  char raw_line[500];
  char sub[100];
  unsigned int sub_idx;
  double tmp_param;
  Eigen::Matrix<float, 3, 4, Eigen::RowMajor> tmp_pose;

  // open gt pose file
  pose_file.open(pose_txt, std::ios::in);
  if (!pose_file.is_open()){
    std::cout << "open gt pose file failed: " << pose_txt << std::endl;
    exit(-1);
  } else {
    // read the poses, for each line(frame)
    for (unsigned int i = 0; i < num_frame; i++){
      pose_file.getline(raw_line, 500);
      if (pose_file.fail()) {
        std::cout << "read line failed!" << std::endl;
        pose_file.close();
        exit(-1);
      }
      scaner = 0;
      // for all parameters of the line, in total 12
      for (int param_i = 0; param_i < 12; param_i++){
        // for each parameter
        sub_idx = 0;
        while (raw_line[scaner] != ' ' && raw_line[scaner] != '\0'){
          sub[sub_idx] = raw_line[scaner];
          scaner++;
          sub_idx++;
        }
        sub[sub_idx] = '\0';
        tmp_param = std::atof(sub);
        tmp_pose(param_i) = float(tmp_param); // assume row-major
        scaner++;
      } // current param
      gt_poses[i] = tmp_pose;
    } // current line
  }
  std::cout << "Read gt poses done for " << num_frame << " frames" << std::endl;
}

void load_data(const std::string& folder_name, std::vector<cv::Mat> &gray, int frame_id){
  std::string left_path = folder_name + "/dataset/sequences/00/image_0/";
  std::string right_path = folder_name + "/dataset/sequences/00/image_1/";
  std::string img_path0 = left_path + std::string(6-std::to_string(frame_id).length(), '0') + std::to_string(frame_id) + ".png";
  std::string img_path1 = right_path + std::string(6-std::to_string(frame_id).length(), '0') + std::to_string(frame_id) + ".png";
  cv::Mat gray_8u;

  std::cout << "reading frame: " << img_path0 << std::endl;
  gray_8u = cv::imread(img_path0, cv::IMREAD_GRAYSCALE);
  if (gray_8u.empty()){
    std::cout << "read img failed." << std::endl;
    std::exit(-1);
  }
  // cv::imshow("left_img", gray_8u);
  gray_8u.convertTo(gray[0], PixelType);

  std::cout << "reading frame: " << img_path1 << std::endl;
  gray_8u = cv::imread(img_path1, cv::IMREAD_GRAYSCALE);
  if (gray_8u.empty()){
    std::cout << "read img failed." << std::endl;
    std::exit(-1);
  }
  // cv::imshow("right_img", gray_8u);
  // cv::waitKey(0);
  gray_8u.convertTo(gray[1], PixelType);
}

void eval_pose(const std::vector<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>& gt_poses, const std::vector<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>& pred_poses){

  unsigned int num_frame = 130;
  float trans_err;
  float sum_err = 0.0f;
  for (unsigned int i = 0; i < num_frame; i++){
    trans_err = (pred_poses[i].block<3,1>(0,3) - gt_poses[i].block<3,1>(0,3)).norm();
    sum_err += trans_err;
    std::cout << "frame " << i << ": " << trans_err << std::endl;
  }
  std::cout << "avg error over " << num_frame << " frames: " << sum_err / float(num_frame) << std::endl;
}

void save_txt(const std::vector<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>& gt_poses, const std::vector<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>& pred_poses){

  // for evaluation script, the size of gt_pose and pred_poses must be the same
  if (gt_poses.size() != pred_poses.size()){
    std::cout << "gt_pose.size() != pred_pose.size()" << std::endl;
    std::cout << "gt_pose size: " << gt_poses.size() << std::endl;
    std::cout << "pred_pose size: " << pred_poses.size() << std::endl;
    exit(-1);
  }

  unsigned int num_frame = gt_poses.size();
  std::string write_gt_file = "../dataset/kitti/devkit/data/odometry/poses/00.txt";
  std::string write_pred_file = "../dataset/kitti/devkit/results/seq00/data/00.txt";
  std::ofstream gt_file;
  std::ofstream pred_file;
  float tmp;
  std::string gt_pose_line;
  std::string pred_pose_line;
  Eigen::Matrix<float, 3, 4, Eigen::RowMajor> gt_pose;
  Eigen::Matrix<float, 3, 4, Eigen::RowMajor> pred_pose;

  // open file & clear all data
  gt_file.open(write_gt_file, std::ios::out | std::ios::trunc);
  if (!gt_file.is_open()){
    std::cout << "open gt write file failed: " << write_gt_file << std::endl;
    exit(-1);
  }
  pred_file.open(write_pred_file, std::ios::out | std::ios::trunc);
  if (!pred_file.is_open()){
    std::cout << "open pred write file failed: " << write_pred_file << std::endl;
    exit(-1);
  }

  // write poses to txt
  for (unsigned int i = 0; i < num_frame; i++){
    gt_pose = gt_poses[i];
    pred_pose = pred_poses[i];
    gt_pose_line = "";
    pred_pose_line = "";
    for (int param_id = 0; param_id < 12; param_id++){
      tmp = gt_pose(param_id);
      gt_pose_line += std::to_string(tmp);
      tmp = pred_pose(param_id);
      pred_pose_line += std::to_string(tmp);
      if (param_id != 11){
        gt_pose_line += " ";
        pred_pose_line += " ";
      }
    }
    gt_file << gt_pose_line << std::endl;
    pred_file << pred_pose_line << std::endl;
  }

  gt_file.close();
  pred_file.close();
  std::cout << "save completed." << std::endl;
}

void save_to_vis(const std::vector<std::tuple<odometry::ImagePyramid, odometry::DepthPyramid, cv::Mat>>& data_vec, const std::vector<int>& keyframe_ids){
  unsigned int num = data_vec.size();
  std::string path_img = "../data/kitti_result/gray_img_left/";
  std::string path_disp = "../data/kitti_result/disparity_left/";
  std::string path_mask = "../data/kitti_result/mask_left/";
  std::string path_keyids = "../data/kitti_result/keyframe_ids/keyframe_id.txt";
  cv::Mat save_img, save_disp, save_mask, tmp_dep;
  std::vector<int> compression_params;
  compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);
  for (unsigned int id=0; id < num; id++){
    std::get<0>(data_vec[id]).GetPyramidImage(0).convertTo(save_img, cv::IMREAD_GRAYSCALE);
    cv::imwrite(path_img+std::to_string(id)+".png", save_img, compression_params);
    std::get<2>(data_vec[id]).convertTo(save_mask, CV_16U);
    cv::imwrite(path_mask+std::to_string(id)+".png", save_mask, compression_params);
    // convert to disparity
    tmp_dep = std::get<1>(data_vec[id]).GetPyramidDepth(0);
    save_disp.create(tmp_dep.rows, tmp_dep.cols, CV_16U);
    for (int y = 0; y < tmp_dep.rows; y++){
      for (int x = 0; x < tmp_dep.cols; x++){
        if (std::get<2>(data_vec[id]).at<uint8_t>(y, x) != 0)
          save_disp.at<uint16_t>(y, x) = uint16_t(386.1448f * tmp_dep.at<float>(y, x));
        else
          save_disp.at<uint16_t>(y, x) = 0;
      }
    }
    cv::imwrite(path_disp+std::to_string(id)+".png", save_disp, compression_params);
  }
  // save keyframe ids
  std::ofstream keyids;
  std::string line_data;
  keyids.open(path_keyids, std::ios::out | std::ios::trunc);
  if (!keyids.is_open()){
    std::cout << "open gt write file failed: " << path_keyids << std::endl;
    exit(-1);
  }
  for (int i=0; i<keyframe_ids.size(); i++){
    line_data = std::to_string(keyframe_ids[i]);
    keyids << line_data << std::endl;
  }
  keyids.close();
}