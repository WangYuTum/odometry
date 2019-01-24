// The file tests camera related operations
// Created by Yu Wang on 2019-01-24.

#include <iostream>
#include <opencv2/core.hpp>
#include "data_types.h"
#include "camera.h"
#include <typeinfo>

int main(){
  // setup camera
  std::string stereo_file = "../calibration_file/camchain.yaml";
  std::shared_ptr<odometry::CameraPyramid> cam_ptr_left, cam_ptr_right;
  cv::Rect valid_region;
  double baseline, left_right_translate; // units are in [meter]
  const int levels = 4;

  odometry::GlobalStatus setup_camera_status=-1;
  setup_camera_status = odometry::SetUpStereoCameraSystem(stereo_file, levels, cam_ptr_left, cam_ptr_right, valid_region, baseline);
  left_right_translate = - baseline;
  if (setup_camera_status == -1){
    std::cout << "Configure stereo camera system failed!" << std::endl;
    exit(-1);
  }
  std::cout << "******************** Before rectification *******************" << std::endl;
  std::cout << "Cam left:" << std::endl;
  std::cout << cam_ptr_left->get_intrinsic_raw() << std::endl;
  std::cout << "distortion coeff: " << std::endl << cam_ptr_left->get_distortion_coeff() << std::endl;
  std::cout << std::endl;

  std::cout << "Cam right:" << std::endl;
  std::cout << cam_ptr_right->get_intrinsic_raw() << std::endl;
  std::cout << "distortion coeff: " << std::endl << cam_ptr_right->get_distortion_coeff() << std::endl;
  std::cout << std::endl;

  std::cout << "******************** After rectification *******************" << std::endl;
  std::cout << "Cam left: " << std::endl;
  std::cout << cam_ptr_left->get_intrinsic_rectified(0) << std::endl;
  std::cout << "sensor_w: " << cam_ptr_left->sensor_w_double() << std::endl;
  std::cout << "sensor_h: " << cam_ptr_left->sensor_h_double() << std::endl;
  std::cout << "pixels_per_mm_x: " << cam_ptr_left->pixels_per_mm_x_double() << std::endl;
  std::cout << "pixels_per_mm_y: " << cam_ptr_left->pixels_per_mm_y_double() << std::endl;
  std::cout << "resolution_raw_w: " << cam_ptr_left->resolution_raw_w() << std::endl;
  std::cout << "resolution_raw_h: " << cam_ptr_left->resolution_raw_h() << std::endl;
  std::cout << std::endl;

  std::cout << "Cam right: " << std::endl << cam_ptr_right->get_intrinsic_rectified(0) << std::endl;
  std::cout << "sensor_w: " << cam_ptr_right->sensor_w_double() << std::endl;
  std::cout << "sensor_h: " << cam_ptr_right->sensor_h_double() << std::endl;
  std::cout << "pixels_per_mm_x: " << cam_ptr_right->pixels_per_mm_x_double() << std::endl;
  std::cout << "pixels_per_mm_y: " << cam_ptr_right->pixels_per_mm_y_double() << std::endl;
  std::cout << "resolution_raw_w: " << cam_ptr_right->resolution_raw_w() << std::endl;
  std::cout << "resolution_raw_h: " << cam_ptr_right->resolution_raw_h() << std::endl;
  std::cout << std::endl;

  std::cout << "******************** Rectified Stereo *******************" << std::endl;
  std::cout << "New baseline: " << baseline << std::endl;
  std::cout << "Valid image region: " << valid_region << std::endl;

  return 0;

}

