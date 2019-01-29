// The file contains (stereo) camera configuration implementations
// Created by Yu Wang on 2019-01-17.

#include <camera.h>
#include <fstream>
#include <stdlib.h>
#include <string>

namespace odometry
{

CameraPyramid::CameraPyramid(int levels, double fx, double fy, double f_theta, double cx, double cy,
                             double k1, double k2, double r1, double r2, double sensor_width, double sensor_height,
                             int resolution_width, int resolution_height){
  levels_ = levels;
  intrinsic_raw_.create(3, 3, CV_64F);
  intrinsic_raw_.at<double>(0, 0) = fx;
  intrinsic_raw_.at<double>(1, 1) = fy;
  intrinsic_raw_.at<double>(0, 2) = cx;
  intrinsic_raw_.at<double>(1, 2) = cy;
  intrinsic_raw_.at<double>(2, 2) = 1;
  intrinsic_raw_.at<double>(0, 1) = f_theta;
  intrinsic_raw_.at<double>(2, 0) = 0;
  intrinsic_raw_.at<double>(1, 0) = 0;
  intrinsic_raw_.at<double>(2, 1) = 0;
  distortion_param_.create(1, 4, CV_64F);
  distortion_param_.at<double>(0, 0) = k1;
  distortion_param_.at<double>(0, 1) = k2;
  distortion_param_.at<double>(0, 2) = r1;
  distortion_param_.at<double>(0, 3) = r2;
  sensor_width_ = sensor_width;
  sensor_height_ = sensor_height;
  resolution_width_ = resolution_width;
  resolution_height_ = resolution_height;
  pixels_per_mm_x_ = resolution_width / sensor_width;
  pixels_per_mm_y_ = resolution_height / sensor_height;
  if (resolution_width_ != 640 || resolution_height_ != 480){
    std::cout << "resolution_width_ != 640 OR resolution_height_ != 480 for creating camera instance!" << std::endl;
    exit(-1);
  }
}

void CameraPyramid::ConfigureCamera(const cv::Mat& rectify_rotation, const cv::Mat& new_intrinsic34,
                     const cv::Size& new_size, int map_type=CV_32FC1, bool use_int_map=false){

  // build pyramid using new intrinsics
  double fx = new_intrinsic34.at<double>(0,0);
  double fy = new_intrinsic34.at<double>(1,1);
  double cx = new_intrinsic34.at<double>(0,2);
  double cy = new_intrinsic34.at<double>(1,2);
  double f_theta = new_intrinsic34.at<double>(0,1);
  for (int l = 0; l < levels_; l++){
    cv::Mat tmp(3, 3, CV_64F);
    tmp.at<double>(0, 0) = fx;
    tmp.at<double>(1, 1) = fy;
    tmp.at<double>(0, 2) = cx;
    tmp.at<double>(1, 2) = cy;
    tmp.at<double>(2, 2) = 1;
    tmp.at<double>(0, 1) = f_theta;
    tmp.at<double>(2, 0) = 0;
    tmp.at<double>(1, 0) = 0;
    tmp.at<double>(2, 1) = 0;
    intrinsic_.emplace_back(tmp);
    fx = fx / 2.0;
    fy = fy / 2.0;
    f_theta = f_theta / 2.0;
    cx = (cx + 0.5) / 2.0 + 0.5;
    cy = (cy + 0.5) / 2.0 + 0.5;
  }
  // get remaps
  cv::initUndistortRectifyMap(intrinsic_raw_, distortion_param_, rectify_rotation, new_intrinsic34, new_size, map_type, rmap_[0], rmap_[1]);
}

GlobalStatus CameraPyramid::UndistortRectify(const cv::Mat& src_raw, cv::Mat& dst, int interpolation=cv::INTER_LINEAR,
                      int borderMode=cv::BORDER_CONSTANT, const cv::Scalar& borderValue = cv::Scalar()){
  // TODO: check img size
  if (src_raw.rows != 480 || src_raw.cols != 640){
    std::cout << "Input for UndistortRectify() is not 480x640!" << std::endl;
    return -1;
  }
  // undistort & rectify image
  cv::remap(src_raw, dst, rmap_[0], rmap_[1], interpolation, borderMode, borderValue);
  return 0;
}


/******************************************** STEREO CAMERA SETUP ***********************************************/
GlobalStatus SetUpStereoCameraSystem(const std::string& stereo_file, int levels, std::shared_ptr<CameraPyramid>& cam_ptr_left,
                                     std::shared_ptr<CameraPyramid>& cam_ptr_right, cv::Rect& valid_region, double& baseline){
  int resolution_w, resolution_h;
  double fx_left, fy_left, f_theta_left, cx_left, cy_left, k1_left, k2_left, r1_left, r2_left;
  double fx_right, fy_right, f_theta_right, cx_right, cy_right, k1_right, k2_right, r1_right, r2_right;
  double sensor_w_left, sensor_h_left;
  double sensor_w_right, sensor_h_right;
  cv::Mat intrinsic_raw_left(3, 3, CV_64F), intrinsic_raw_right(3, 3, CV_64F);
  cv::Mat dist_left(1, 4, CV_64F), dist_right(1, 4, CV_64F);
  cv::Mat rotate_left_right(3, 3, CV_64F); // rotation of right relative to left
  cv::Mat translate_left_right(3, 1, CV_64F); // translation of right relative to left, should be negative
  cv::Size img_size(640, 480); // (width, height), NOTE the size must be the same during calibration, stereoRectify and ConfigureCamera
  cv::Mat rectify_rotate_left, rectify_rotate_right;
  cv::Mat intrinsic_left_new, intrinsic_right_new; // the new intrinsics 3x4
  cv::Mat disp_to_depth; // transform from disparity to depth 4x4

  // valid region after rectification;
  cv::Rect validRoi_left, validRoi_right; // public attributes: x-top_left (inclusive), y-top_left (inclusive), height(exclusive), width(exclusive)
  int top_left_x, top_left_y, bot_right_x, bot_right_y, height, width;
  // read stereo system parameters from calibration file, assign values
  ReadStereoCalibrationFile(stereo_file, intrinsic_raw_left, intrinsic_raw_right, dist_left, dist_right, rotate_left_right, translate_left_right,
                            sensor_w_left, sensor_h_left, sensor_w_right, sensor_h_right, resolution_w, resolution_h);
  fx_left = intrinsic_raw_left.at<double>(0,0);
  fy_left = intrinsic_raw_left.at<double>(1,1);
  f_theta_left = intrinsic_raw_left.at<double>(0,1);
  cx_left = intrinsic_raw_left.at<double>(0,2);
  cy_left = intrinsic_raw_left.at<double>(1,2);
  k1_left = dist_left.at<double>(0,0);
  k2_left = dist_left.at<double>(0,1);
  r1_left = dist_left.at<double>(0,2);
  r2_left = dist_left.at<double>(0,3);

  fx_right = intrinsic_raw_right.at<double>(0,0);
  fy_right = intrinsic_raw_right.at<double>(1,1);
  f_theta_right = intrinsic_raw_right.at<double>(0,1);
  cx_right = intrinsic_raw_right.at<double>(0,2);
  cy_right = intrinsic_raw_right.at<double>(1,2);
  k1_right = dist_right.at<double>(0,0);
  k2_right = dist_right.at<double>(0,1);
  r1_right = dist_right.at<double>(0,2);
  r2_right = dist_right.at<double>(0,3);
//  std::cout << "camera left: " << std::endl <<  intrinsic_raw_left << std::endl;
//  std::cout << "camera right: " << std::endl <<  intrinsic_raw_left << std::endl;
//  std::cout << "distortion left: " << std::endl << dist_left << std::endl;
//  std::cout << "distortion right: " << std::endl << dist_right << std::endl;
//  std::cout << "relative rotation: " << std::endl << rotate_left_right << std::endl;
//  std::cout << "relative translation: " << std::endl << translate_left_right << std::endl;

  // create camera instances
  cam_ptr_left = std::make_shared<CameraPyramid>(levels, fx_left, fy_left, f_theta_left, cx_left, cy_left,
                                                 k1_left, k2_left, r1_left, r2_left, sensor_w_left, sensor_h_left, resolution_w, resolution_h);
  cam_ptr_right = std::make_shared<CameraPyramid>(levels, fx_right, fy_right, f_theta_right, cx_right, cy_right,
                                                 k1_right, k2_right, r1_right, r2_right, sensor_w_right, sensor_h_right, resolution_w, resolution_h);
  // stereo rectify
//  std::cout << "rectifing cameras..." << std::endl;
  cv::stereoRectify(intrinsic_raw_left, dist_left, intrinsic_raw_right, dist_right, img_size, rotate_left_right, translate_left_right,
                    rectify_rotate_left, rectify_rotate_right, intrinsic_left_new, intrinsic_right_new, disp_to_depth,
                    cv::CALIB_ZERO_DISPARITY, 1, img_size, &validRoi_left, &validRoi_right);
//  std::cout << "rectify done." << std::endl;
  // get the new baseline
  baseline = std::fabs(intrinsic_right_new.at<double>(0,3) / intrinsic_right_new.at<double>(0,0));
//  std::cout << "new baseline: " << baseline << std::endl;
//  std::cout << "new camera left: " << std::endl <<  intrinsic_left_new << std::endl;
//  std::cout << "new camera right: " << std::endl <<  intrinsic_right_new << std::endl;

  cam_ptr_left->ConfigureCamera(rectify_rotate_left, intrinsic_left_new, img_size);
  cam_ptr_right->ConfigureCamera(rectify_rotate_right, intrinsic_right_new, img_size);
  // check valid regions
  top_left_x = (validRoi_left.x >= validRoi_right.x) ? validRoi_left.x : validRoi_right.x;
  top_left_y = (validRoi_left.y >= validRoi_right.y) ? validRoi_left.y : validRoi_right.y;
  bot_right_x = (validRoi_left.x+validRoi_left.width <= validRoi_right.x+validRoi_right.width) ? validRoi_left.x+validRoi_left.width : validRoi_right.x+validRoi_right.width;
  bot_right_y = (validRoi_left.y+validRoi_left.height <= validRoi_right.y+validRoi_right.height) ? validRoi_left.y+validRoi_left.height : validRoi_right.y+validRoi_right.height;
  height = bot_right_y - top_left_y;
  width = bot_right_x - top_left_x;
  valid_region = cv::Rect(top_left_x, top_left_y, width, height);
//  std::cout << "new valid image region: " << std::endl << valid_region << std::endl;
  std::cout << "stereo configuration done!" << std::endl;

  // succeed
  return 0;
}

void ReadStereoCalibrationFile(const std::string& stereo_file, cv::Mat& camera_intrin_left, cv::Mat& camera_intrin_right,
        cv::Mat& dist_left, cv::Mat& dist_right, cv::Mat& rotate_left_right, cv::Mat& translate_left_right,
        double& sensor_w_left, double& sensor_h_left, double& sensor_w_right, double& sensor_h_right, int& resolution_w, int& resolution_h){
  // Params(FP-64bit):
  //  * camera_intrin_left: [3,3]
  //  * camera_intrin_right: [3,3]
  //  * dist_left: [1,4]
  //  * dist_right: [1,4]
  //  * rotate_left_right: [3,3]
  //  * translate_left_right: [3,1]

  std::ifstream file;
  char raw_line[500];
  std::string line_string;

  // define data
  double distortion0[4], distortion1[4];
  double intrinsic0[4], intrinsic1[4];
  double extrinsic[16];
  double sensor_left[2], sensor_right[2]; // [width, height] in mm
  int resolution[2]; // [width, height] in pixels
  double* dist_ptr = nullptr;
  double* intrin_ptr = nullptr;
  double* extrin_ptr = extrinsic;
  double* sensor_ptr = nullptr;
  int* resolution_ptr = resolution;
  int start, scaner;
  char sub[50];

  std::cout << "Reading calibration file: " << stereo_file << std::endl;
  file.open(stereo_file, std::ios::in);
  if (!file.is_open()){
    std::cout << "open calibration file failed!" << std::endl;
    exit(-1);
  } else{
    while (file.getline(raw_line, 500)){
      if (file.fail()) {
        std::cout << "read line failed!" << std::endl;
        file.close();
        exit(-1);
      }
      // set the correct data pointer
      if (std::string(raw_line) == "cam0:"){
        dist_ptr = distortion0;
        intrin_ptr = intrinsic0;
        sensor_ptr = sensor_left;
      } else if (std::string(raw_line) == "cam1:"){
        dist_ptr = distortion1;
        intrin_ptr = intrinsic1;
        sensor_ptr = sensor_right;
      }
      // parse data
      if (std::string(raw_line).find("distortion_coeffs:") != std::string::npos){
        start = int(std::string(raw_line).find("distortion_coeffs:")) + 20;
        scaner = start;
        for (int i = 0; i < 4; i++){
          int sub_idx = 0;
          while (raw_line[scaner] != ',' && raw_line[scaner] != ']'){
            sub[sub_idx] = raw_line[scaner];
            scaner++;
            sub_idx++;
          }
          sub[sub_idx] = '\0';
          *(dist_ptr+i) = std::atof(sub);
          sub[0] = '\0';
          scaner = scaner + 2;
        }
      } else if (std::string(raw_line).find("intrinsics:") != std::string::npos){
        start = int(std::string(raw_line).find("intrinsics:")) + 13;
        scaner = start;
        for (int i = 0; i < 4; i++){
          int sub_idx = 0;
          while (raw_line[scaner] != ',' && raw_line[scaner] != ']'){
            sub[sub_idx] = raw_line[scaner];
            scaner++;
            sub_idx++;
          }
          sub[sub_idx] = '\0';
          *(intrin_ptr+i) = std::atof(sub);
          sub[0] = '\0';
          scaner = scaner + 2;
        }
      } else if (std::string(raw_line).find("T_cn_cnm1:") != std::string::npos){
        // read the following 4 lines to get extrinsics
        for (int line_idx = 0; line_idx < 4; line_idx++){
          file.getline(raw_line, 500);
          if (file.fail()) {
            std::cout << "read line failed!" << std::endl;
            file.close();
            exit(-1);
          }
          start = int(std::string(raw_line).find("- [")) + 3;
          scaner = start;
          for (int i = 0; i < 4; i++){
            int sub_idx = 0;
            while (raw_line[scaner] != ',' && raw_line[scaner] != ']'){
              sub[sub_idx] = raw_line[scaner];
              scaner++;
              sub_idx++;
            }
            sub[sub_idx] = '\0';
            *(extrin_ptr+i) = std::atof(sub);
            sub[0] = '\0';
            scaner = scaner + 2;
          }
          extrin_ptr += 4;
        }
      } else if (std::string(raw_line).find("sensor_size:") != std::string::npos){
        start = int(std::string(raw_line).find("sensor_size:")) + 14;
        scaner = start;
        for (int i = 0; i < 2; i++){
          int sub_idx = 0;
          while (raw_line[scaner] != ',' && raw_line[scaner] != ']'){
            sub[sub_idx] = raw_line[scaner];
            scaner++;
            sub_idx++;
          }
          sub[sub_idx] = '\0';
          *(sensor_ptr+i) = std::atof(sub);
          sub[0] = '\0';
          scaner = scaner + 2;
        }
      } else if (std::string(raw_line).find("resolution:") != std::string::npos){
        start = int(std::string(raw_line).find("resolution:")) + 13;
        scaner = start;
        for (int i = 0; i < 2; i++){
          int sub_idx = 0;
          while (raw_line[scaner] != ',' && raw_line[scaner] != ']'){
            sub[sub_idx] = raw_line[scaner];
            scaner++;
            sub_idx++;
          }
          sub[sub_idx] = '\0';
          *(resolution_ptr+i) = std::atoi(sub);
          sub[0] = '\0';
          scaner = scaner + 2;
        }
      }
    }
  }
  file.close();
  std::cout << "Read completed. File closed." << std::endl;
  // assign distortion coeff
  for (int i = 0; i < 4; i++) {
    dist_left.at<double>(0, i) = distortion0[i];
    dist_right.at<double>(0, i) = distortion1[i];
  }
  // assign intrinsics
  camera_intrin_left.setTo(cv::Scalar(0.0));
  camera_intrin_left.at<double>(0,0) = intrinsic0[0];
  camera_intrin_left.at<double>(1,1) = intrinsic0[1];
  camera_intrin_left.at<double>(0,2) = intrinsic0[2];
  camera_intrin_left.at<double>(1,2) = intrinsic0[3];
  camera_intrin_left.at<double>(2,2) = 1.0;

  camera_intrin_right.setTo(cv::Scalar(0));
  camera_intrin_right.at<double>(0,0) = intrinsic1[0];
  camera_intrin_right.at<double>(1,1) = intrinsic1[1];
  camera_intrin_right.at<double>(0,2) = intrinsic1[2];
  camera_intrin_right.at<double>(1,2) = intrinsic1[3];
  camera_intrin_right.at<double>(2,2) = 1.0;

  // assign extrinsics
  rotate_left_right.at<double>(0,0) = extrinsic[0];
  rotate_left_right.at<double>(0,1) = extrinsic[1];
  rotate_left_right.at<double>(0,2) = extrinsic[2];
  translate_left_right.at<double>(0,0) = extrinsic[3];
  rotate_left_right.at<double>(1,0) = extrinsic[4];
  rotate_left_right.at<double>(1,1) = extrinsic[5];
  rotate_left_right.at<double>(1,2) = extrinsic[6];
  translate_left_right.at<double>(1,0) = extrinsic[7];
  rotate_left_right.at<double>(2,0) = extrinsic[8];
  rotate_left_right.at<double>(2,1) = extrinsic[9];
  rotate_left_right.at<double>(2,2) = extrinsic[10];
  translate_left_right.at<double>(2,0) = extrinsic[11];

  // assign sensor params
  sensor_w_left = sensor_left[0];
  sensor_h_left = sensor_left[1];
  sensor_w_right = sensor_right[0];
  sensor_h_right = sensor_right[1];
  resolution_w = resolution[0];
  resolution_h = resolution[1];

}

} // namespace odometry

