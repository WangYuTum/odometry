// The file contains (stereo) camera configuration implementations
// Created by Yu Wang on 2019-01-17.

#include <camera.h>

namespace odometry
{

CameraPyramid::CameraPyramid(int levels, float fx, float fy, float f_theta, float cx, float cy,
        float k1, float k2, float r1, float r2, float sensor_width, float sensor_height,
        float resolution_width, float resolution_height){
  levels_ = levels;
  intrinsic_raw_.create(3, 3, CV_32F);
  intrinsic_raw_.at<float>(0, 0) = fx;
  intrinsic_raw_.at<float>(1, 1) = fy;
  intrinsic_raw_.at<float>(0, 2) = cx;
  intrinsic_raw_.at<float>(1, 2) = cy;
  intrinsic_raw_.at<float>(2, 2) = 1;
  intrinsic_raw_.at<float>(0, 1) = f_theta;
  intrinsic_raw_.at<float>(2, 0) = 0;
  intrinsic_raw_.at<float>(1, 0) = 0;
  intrinsic_raw_.at<float>(2, 1) = 0;
  distortion_param_.create(1, 4, CV_32F);
  distortion_param_.at<float>(0, 0) = k1;
  distortion_param_.at<float>(0, 1) = k2;
  distortion_param_.at<float>(0, 2) = r1;
  distortion_param_.at<float>(0, 3) = r2;
  sensor_width_ = sensor_width;
  sensor_height_ = sensor_height;
  resolution_width_ = resolution_width;
  resolution_height_ = resolution_height;
  pixels_per_mm_x_ = resolution_width / sensor_width;
  pixels_per_mm_y_ = resolution_height / sensor_height;
}

void CameraPyramid::ConfigureCamera(const cv::Mat& rectify_rotation, const cv::Mat& new_intrinsic34,
                     const cv::Size& new_size, int map_type=CV_32FC1, bool use_int_map=false){

  // build pyramid using new intrinsics
  float fx = new_intrinsic34.at<float>(0,0);
  float fy = new_intrinsic34.at<float>(1,1);
  float cx = new_intrinsic34.at<float>(0,2);
  float cy = new_intrinsic34.at<float>(1,2);
  float f_theta = new_intrinsic34.at<float>(0,1);
  for (int l = 0; l < levels_; l++){
    cv::Mat tmp(3, 3, CV_32F);
    tmp.at<float>(0, 0) = fx;
    tmp.at<float>(1, 1) = fy;
    tmp.at<float>(0, 2) = cx;
    tmp.at<float>(1, 2) = cy;
    tmp.at<float>(2, 2) = 1;
    tmp.at<float>(0, 1) = f_theta;
    tmp.at<float>(2, 0) = 0;
    tmp.at<float>(1, 0) = 0;
    tmp.at<float>(2, 1) = 0;
    intrinsic_.emplace_back(tmp);
    fx = fx / 2.0f;
    fy = fy / 2.0f;
    f_theta = f_theta / 2.0f;
    cx = (cx + 0.5f) / 2.0f + 0.5f;
    cy = (cy + 0.5f) / 2.0f + 0.5f;
  }
  // get remaps
  cv::initUndistortRectifyMap(intrinsic_raw_, distortion_param_, rectify_rotation, new_intrinsic34, new_size, map_type, rmap_[0], rmap_[1]);
  // TODO: convert float-remap to int-remap
}

GlobalStatus CameraPyramid::UndistortRectify(const cv::Mat& src_raw, cv::Mat& dst, int interpolation=cv::INTER_LINEAR,
                      int borderMode=cv::BORDER_CONSTANT, const cv::Scalar& borderValue = cv::Scalar()){
  // check src
  if (src_raw.rows != 480 || src_raw.cols != 640){
    std::cout << "camera raw image is not 480x640!" << std::endl;
    return -1;
  }
  // undistort & rectify image
  cv::remap(src_raw, dst, rmap_[0], rmap_[1], interpolation, borderMode, borderValue);
  return 0;
}


/******************************************** STEREO CAMERA SETUP ***********************************************/
GlobalStatus SetUpStereoCameraSystem(const std::string& stereo_file, std::shared_ptr<CameraPyramid> cam_ptr_left,
                                     std::shared_ptr<CameraPyramid> cam_ptr_right, cv::Rect& valid_region){
  float levels;
  float fx_left, fy_left, f_theta_left, cx_left, cy_left, k1_left, k2_left, r1_left, r2_left;
  float fx_right, fy_right, f_theta_right, cx_right, cy_right, k1_right, k2_right, r1_right, r2_right;
  float sensor_w_left, sensor_h_left;
  float sensor_w_right, sensor_h_right;
  float resolution_w, resolution_h;
  cv::Mat intrinsic_raw_left(3, 3, CV_32F), intrinsic_raw_right(3, 3, CV_32F);
  cv::Mat dist_left(1, 4, CV_32F), dist_right(1, 4, CV_32F);
  cv::Mat rotate_left_right(3, 3, CV_32F); // rotation of right relative to left
  cv::Mat translate_left_right(3, 1, CV_32F); // translation of right relative to left
  cv::Size img_size(480, 640); // (rows, cols)
  cv::Mat rectify_rotate_left, rectify_rotate_right;
  cv::Mat intrinsic_left_new, intrinsic_right_new; // the new intrinsics 3x4
  cv::Mat disp_to_depth; // transform from disparity to depth 4x4

  // valid region after rectification;
  cv::Rect validRoi_left, validRoi_right; // public attributes: x-top_left (inclusive), y-top_left (inclusive), height(exclusive), width(exclusive)
  int top_left_x, top_left_y, height, width;
  // TODO: read stereo system parameters


  // create camera instances
  cam_ptr_left = std::make_shared<CameraPyramid>(levels, fx_left, fy_left, f_theta_left, cx_left, cy_left,
                                                 k1_left, k2_left, r1_left, r2_left, sensor_w_left, sensor_h_left, resolution_w, resolution_h);
  cam_ptr_right = std::make_shared<CameraPyramid>(levels, fx_right, fy_right, f_theta_right, cx_right, cy_right,
                                                 k1_right, k2_right, r1_right, r2_right, sensor_w_right, sensor_h_right, resolution_w, resolution_h);
  // stereo rectify
  cv::stereoRectify(intrinsic_raw_left, dist_left, intrinsic_raw_right, dist_right, img_size, rotate_left_right, translate_left_right,
                    rectify_rotate_left, rectify_rotate_right, intrinsic_left_new, intrinsic_right_new, disp_to_depth,
                    cv::CALIB_ZERO_DISPARITY, 1, img_size, &validRoi_left, &validRoi_right);
  // setup left/right cameras
  cam_ptr_left->ConfigureCamera(rectify_rotate_left, intrinsic_left_new, img_size);
  cam_ptr_right->ConfigureCamera(rectify_rotate_right, intrinsic_right_new, img_size);
  // check valid regions
  top_left_x = (validRoi_left.x >= validRoi_right.x) ? validRoi_left.x : validRoi_right.x;
  top_left_y = (validRoi_left.y >= validRoi_right.y) ? validRoi_left.y : validRoi_right.y;
  height = (validRoi_left.x+validRoi_left.height <= validRoi_right.x+validRoi_right.height) ? validRoi_left.x+validRoi_left.height : validRoi_right.x+validRoi_right.height;
  width = (validRoi_left.x+validRoi_left.width <= validRoi_right.x+validRoi_right.width) ? validRoi_left.x+validRoi_left.width : validRoi_right.x+validRoi_right.width;
  valid_region = cv::Rect(top_left_x, top_left_y, width, height);

  // succeed
  return 0;
}

} // namespace odometry

