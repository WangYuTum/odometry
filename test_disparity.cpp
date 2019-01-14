// The file is used to test disparity search and depth estimation
// Created by Yu Wang on 2019-01-14.

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <fstream>
#include "data_types.h"
#include <depth_estimate.h>
#include <typeinfo>
#include <se3.hpp>

const std::string kDataPath = "../dataset/disparity_cones";
//const std::string kDataPath = "../dataset/disparity_teddy";

void load_data(const std::string& folder_name, std::vector<cv::Mat> &gray, std::vector<cv::Mat> &disp);
void report_error(const cv::Mat& pred_disp, const cv::Mat& gt_disp, const cv::Mat& valid_map);

int main() {

  // load data
  std::vector<cv::Mat> gray(2); // float intensity
  std::vector<cv::Mat> disp(2); // float disp
  load_data(kDataPath, gray, disp);

  // create depth estimator
  odometry::GlobalStatus depth_state;
  odometry::DepthEstimator depth_est(65.0f, 2000.0f);
  cv::Scalar init_val(0);
  cv::Mat left_val(gray[0].rows, gray[0].cols, CV_8U, init_val);
  cv::Mat left_disp(gray[0].rows, gray[0].cols, PixelType);
  cv::Mat left_dep(gray[0].rows, gray[0].cols, PixelType);
  std::cout << "call compute." << std::endl;
  depth_state = depth_est.ComputeDepth(gray[0], gray[1], left_val, left_disp, left_dep);
  if (depth_state != -1){
    std::cout << "compute disparity succeed." << std::endl;
    std::cout << "number of val disparity: " << cv::sum(left_val)[0] << std::endl;
    // report error
    report_error(left_disp, disp[0], left_val);
    cv::Mat pred_disp;
    cv::Mat gt_disp;
    cv::Mat valid_map;
    left_disp.convertTo(pred_disp, cv::IMREAD_GRAYSCALE, 4.0f);
    disp[0].convertTo(gt_disp, cv::IMREAD_GRAYSCALE, 4.0f);
    left_val.convertTo(valid_map, cv::IMREAD_GRAYSCALE, 255.0f);
    cv::imshow("computed disp", pred_disp);
    cv::imshow("gt disp", gt_disp);
    cv::imshow("valid map", valid_map);
    cv::waitKey(0);
  } else {
    std::cout << "compute disparity failed." << std::endl;
  }
  return 0;
}

void load_data(const std::string& folder_name, std::vector<cv::Mat> &gray, std::vector<cv::Mat> &disp){
  std::string left_img_file = folder_name + "/im2.png";
  std::string right_img_file = folder_name + "/im6.png";
  std::string left_disp_file = folder_name + "/disp2.png";
  std::string right_disp_file = folder_name + "/disp6.png";
  cv::Mat gray_8u;

  // read left image
  gray_8u = cv::imread(left_img_file, cv::IMREAD_GRAYSCALE);
  if (gray_8u.empty()){
    std::cout << "read left img failed." << std::endl;
    std::exit(-1);
  }
  //cv::imshow("left_img", gray_8u);
  gray_8u.convertTo(gray[0], PixelType);

  // read right image
  gray_8u = cv::imread(right_img_file, cv::IMREAD_GRAYSCALE);
  if (gray_8u.empty()){
    std::cout << "read right img failed." << std::endl;
    std::exit(-1);
  }
  //cv::imshow("right_img", gray_8u);
  gray_8u.convertTo(gray[1], PixelType);

  // read left disp
  gray_8u = cv::imread(left_disp_file, cv::IMREAD_GRAYSCALE);
  if (gray_8u.empty()){
    std::cout << "read left disp failed." << std::endl;
    std::exit(-1);
  }
  //cv::imshow("left_disp", gray_8u);
  gray_8u.convertTo(disp[0], PixelType, 1.0f/4.0f);

  // read right disp
  gray_8u = cv::imread(right_disp_file, cv::IMREAD_GRAYSCALE);
  if (gray_8u.empty()){
    std::cout << "read left disp failed." << std::endl;
    std::exit(-1);
  }
  //cv::imshow("right_disp", gray_8u);
  gray_8u.convertTo(disp[1], PixelType, 1.0f/4.0f);
  //cv::waitKey(0);
}

void report_error(const cv::Mat& pred_disp, const cv::Mat& gt_disp, const cv::Mat& valid_map){
  std::vector<int> statistic{0,0,0,0,0,0,0,0,0,0};
  float err_sum = 0;
  float abs_err = 0;
  float sum_val = float(cv::sum(valid_map)[0]);
  for (int y=0; y<375; y++){
    for (int x=0; x<450; x++){
      if (valid_map.at<uint8_t>(y,x) == 1){
        abs_err = std::abs(gt_disp.at<float>(y,x) - pred_disp.at<float>(y,x));
        if (abs_err == 0.0) statistic[0] += 1;
        else if (abs_err == 1.0) statistic[1] += 1;
        else if (abs_err == 2.0) statistic[2] += 1;
        else if (abs_err == 3.0) statistic[3] += 1;
        else if (abs_err == 4.0) statistic[4] += 1;
        else if (abs_err == 5.0) statistic[5] += 1;
        else if (abs_err <= 7.0) statistic[6] += 1;
        else if (abs_err <= 10.0) statistic[7] += 1;
        else if (abs_err <= 20.0) statistic[8] += 1;
        else statistic[9] += 1;
        err_sum += abs_err;
      } else
        continue;
    }
  }
  std::cout << "Error = 0 : " << statistic[0] << ", percentage: " << float(statistic[0])/sum_val << std::endl;
  std::cout << "Error = 1 : " << statistic[1] << ", percentage: " << float(statistic[1])/sum_val << std::endl;
  std::cout << "Error = 2 : " << statistic[2] << ", percentage: " << float(statistic[2])/sum_val << std::endl;
  std::cout << "Error = 3 : " << statistic[3] << ", percentage: " << float(statistic[3])/sum_val << std::endl;
  std::cout << "Error = 4 : " << statistic[4] << ", percentage: " << float(statistic[4])/sum_val << std::endl;
  std::cout << "Error = 5 : " << statistic[5] << ", percentage: " << float(statistic[5])/sum_val << std::endl;
  std::cout << "Error 5-7 : " << statistic[6] << ", percentage: " << float(statistic[6])/sum_val << std::endl;
  std::cout << "Error 8-10 : " << statistic[7] << ", percentage: " << float(statistic[7])/sum_val << std::endl;
  std::cout << "Error 11-20 : " << statistic[8] << ", percentage: " << float(statistic[8])/sum_val << std::endl;
  std::cout << "Error >20 : " << statistic[9] << ", percentage: " << float(statistic[9])/sum_val << std::endl;
  std::cout << "average pixel error: " << err_sum / sum_val << std::endl;
}