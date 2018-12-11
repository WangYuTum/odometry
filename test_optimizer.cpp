// Test Camera Tracking over a sequence of pairs of consecutive images

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <fstream>
#include "include/camera.h"
#include "include/data_types.h"
#include "include/image_processing_global.h"
#include "include/image_pyramid.h"
#include "include/lm_optimizer.h"
#include <se3.hpp>
#include <typeinfo>

#ifndef N_FRAMES
#define N_FRAMES 10 // 200
#endif
const std::string kDataPath = "../dataset/rgbd_dataset_freiburg3_teddy";

void load_data(std::string filename, std::vector<cv::Mat> &gray, std::vector<cv::Mat> &depth, Eigen::MatrixXf &poses, int n_frames);

int main() {

  /******************************* CREATE CAMERA INSTANCE ***********************************/
  // use Freiburg 3 sequence (already undistorted, rgb and depth are already pre-registered)
  float fx = 535.4, fy = 539.2, f_theta = 0, cx = 320.1, cy = 247.6;
  std::shared_ptr<odometry::Camera> camera_ptr = std::make_shared<odometry::Camera>(fx, fy, f_theta, cx, cy);


  /******************************* LOAD DATASET ***********************************/
  Eigen::MatrixXf poses(7, N_FRAMES); // qw, qx, qy, qz, tx, ty, tz; absolute pose of cameras w.r.t. world origin
  std::vector<cv::Mat> gray(N_FRAMES); // float intensity
  std::vector<cv::Mat> depth(N_FRAMES); // float depth
  load_data(kDataPath + "/associated.txt", gray, depth, poses, N_FRAMES);
  std::cout << "Load data done: " << N_FRAMES << " frames" << std::endl;


  /******************************* CREATE IMAGE/DEPTH PYRAMIDS ***********************************/
  // all input rgb must already be converted to grayscale(float), depth must also be float(in meters, invalid = 0)
  // avoid copying using unique_ptr
  std::vector<std::unique_ptr<odometry::ImagePyramid>> img_pyramids;
  std::vector<std::unique_ptr<odometry::DepthPyramid>> dep_pyramids;
  clock_t begin = clock();
  for (int idx = 0; idx < N_FRAMES; idx++){
    img_pyramids.emplace_back(std::make_unique<odometry::ImagePyramid>(4, gray[idx], false));
    dep_pyramids.emplace_back(std::make_unique<odometry::DepthPyramid>(4, depth[idx], false));
  }
  clock_t end = clock();
  std::cout << "Created Pyramids(gray, depth): " << double(end - begin) / CLOCKS_PER_SEC * 1000.0f / float(N_FRAMES) << " ms/img." << std::endl;


  /******************************* CREATE OPTIMIZER INSTANCE ***********************************/
  std::vector<int> max_iters = {10, 20, 20, 30};
  odometry::Matrix44f init_affine;
  init_affine.block<3,3>(0,0) = Eigen::Matrix<float, 3, 3>::Identity();
  init_affine.block<1,4>(3,0) << 0.0f, 0.0f, 0.0f, 1.0f;
  init_affine.block<3,1>(0,3) << 0.0f, 0.0f, 0.0f;
  odometry::LevenbergMarquardtOptimizer optimizer(0.001f, 5e-7f, max_iters, init_affine, camera_ptr);
  std::cout << "Created optimizer instance." << std::endl;


  /******************************* ESTIMATE POSE ***********************************/
  // optimize relative camera pose of pairs of frames, show statistics
  std::cout << "Start optimizing ..." << std::endl;
  odometry::Matrix44f rela_affine = optimizer.Solve(*img_pyramids[0], *dep_pyramids[0], *img_pyramids[1]);
  std::cout << "pred relative pose: " << std::endl << rela_affine << std::endl;
  Eigen::AngleAxisf rotation_mat_0(Eigen::Quaternionf(poses(0,0), poses(1,0), poses(2,0), poses(3,0)));
  Eigen::AngleAxisf rotation_mat_1(Eigen::Quaternionf(poses(0,1), poses(1,1), poses(2,1), poses(3,1)));
  Eigen::Matrix4f rel_pose;
  odometry::Matrix44f affine1;
  rel_pose.block<3,3>(0,0) = rotation_mat_1.toRotationMatrix().transpose() * rotation_mat_0.toRotationMatrix().transpose();
  rel_pose.block<3,1>(0,3) << poses(4,0) - poses(4,1), poses(5,0) - poses(5,1), poses(6,0) - poses(6,1);
  rel_pose.block<1,4>(3,0) << 0.0f, 0.0f, 0.0f, 1.0f;
  std::cout << "true relative pose: " << std::endl << rel_pose << std::endl;

//  odometry::Matrix44f affine0;
//  Eigen::AngleAxisf rotation_mat_0(Eigen::Quaternionf(poses(0,0), poses(1,0), poses(2,0), poses(3,0)));
//  affine0.block<3,3>(0,0) = rotation_mat_0.toRotationMatrix();
//  affine0.block<1,4>(3,0) << 0.0f, 0.0f, 0.0f, 1.0f;
//  affine0.block<3,1>(0,3) << poses(4,0), poses(5,0), poses(6,0);
//  std::cout << "pose0: " << affine0 << std::endl;
//
//  odometry::Matrix44f affine1;
//  Eigen::AngleAxisf rotation_mat_1(Eigen::Quaternionf(poses(0,1), poses(1,1), poses(2,1), poses(3,1)));
//  affine1.block<3,3>(0,0) = rotation_mat_1.toRotationMatrix();
//  affine1.block<1,4>(3,0) << 0.0f, 0.0f, 0.0f, 1.0f;
//  affine1.block<3,1>(0,3) << poses(4,1), poses(5,1), poses(6,1);
//  std::cout << "pose1: " << affine1 << std::endl;




  /******************************* EVALUATE ***********************************/
  // TODO 5: evaluate accuray as axis angle(3 params) and translation(3 params)

  return 0;
}

void load_data(std::string filename, std::vector<cv::Mat> &gray, std::vector<cv::Mat> &depth, Eigen::MatrixXf &poses, int n_frames) {
  std::string line;
  std::ifstream file(filename);
  int counter = 0;
  if (file.is_open()) {
    while (std::getline(file, line) && counter < n_frames) {
      std::vector<std::string> items;
      std::string item;
      std::stringstream ss(line);
      while (std::getline(ss, item, ' '))
        items.push_back(item);

      // -> load gray
      std::string filename_rgb = std::string(kDataPath + "/") + items[9];
      cv::Mat gray_8u = cv::imread(filename_rgb, cv::IMREAD_GRAYSCALE);
      gray_8u.convertTo(gray[counter], PixelType);
      // <-

      // -> load depth
      std::string filename_depth = std::string(kDataPath + "/") + items[11];
      cv::Mat depth_img = cv::imread(filename_depth, cv::IMREAD_UNCHANGED);
      depth_img.convertTo(depth[counter], PixelType, 1.0f/5000.0f);

      // -> pose
      Eigen::Vector3f t(std::stof(items[1]), std::stof(items[2]), std::stof(items[3])); // <- translation T
      //Eigen::Quaternionf q(std::stof(items[7]), std::stof(items[4]), std::stof(items[5]), std::stof(items[6])); // <- rotation in Eigen: w,x,y,z
      //Eigen::Vector3f a = Eigen::AngleAxisf(q).angle()*Eigen::AngleAxisf(q).axis(); // <-- convert to axis angle
      poses.col(counter) << std::stof(items[7]), std::stof(items[4]), std::stof(items[5]), std::stof(items[6]), t(0), t(1), t(2);
      counter++;
    }
    file.close();
  }
  assert(counter == n_frames);
};