// The file is used to test disparity search and depth estimation assuming given undistorted & rectified image pair
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

const std::string kDataPath = "../dataset/disparity_bowling2_full";

void load_data(const std::string& folder_name, std::vector<cv::Mat> &gray, std::vector<cv::Mat> &disp);
void report_disp_error(const cv::Mat& pred_disp, const cv::Mat& gt_disp, const cv::Mat& valid_map, const cv::Mat& gt_valid_map);
void report_depth_error(const cv::Mat& pred_depth, const cv::Mat& gt_depth, const cv::Mat& valid_map, const cv::Mat& gt_valid_map);

int main() {


  // Dataset calibration
  /*
  fx = fy = 3740 [pixels for full resolution],
  doffs = 240 [pixels for full resolution],
  baseline = 0.16 [meters for full resolution],
  */
  float fx = 3740.0f; // since downsample by 3
  float doffs = 0.0f; // need to be added to disparity when convert to 3d depth， 240
  float baseline = 0.16f; // in meters
  float tmp;
  cv::Scalar init_val(0);

  // load data & create depth map
  std::vector<cv::Mat> gray(2); // float intensity
  std::vector<cv::Mat> gt_disp(2); // float disp, exact value
  load_data(kDataPath, gray, gt_disp);
  cv::Mat gt_depth(gray[0].rows, gray[0].cols, PixelType, init_val);
  cv::Mat gt_valid_map(gray[0].rows, gray[0].cols, CV_8U, init_val);
  float min_depth=100000, max_depth=0;
  for (int y = 0; y < gray[0].rows; y++){
    for (int x = 0; x < gray[0].cols; x++){
      tmp = gt_disp[0].at<float>(y, x);
      if (tmp != 0){
        gt_valid_map.at<uint8_t>(y, x) = 1;
        // depth = fx * baseline / disp
        gt_depth.at<float>(y, x) = (tmp + doffs) / (fx * baseline);
        min_depth = (1.0f / gt_depth.at<float>(y, x) < min_depth)? 1.0f / gt_depth.at<float>(y, x) : min_depth;
        max_depth = (1.0f / gt_depth.at<float>(y, x) > max_depth)? 1.0f / gt_depth.at<float>(y, x) : max_depth;
      }
    }
  }
  std::cout << "created gt inverse depth. min: " << min_depth << " meters, " << "max: " << max_depth << " meters." << std::endl;

  // estimate depth
  //   DepthEstimator(float grad_th, float ssd_th, float photo_th, float min_depth, float max_depth, float lambda, float huber_delta,
  //          float precision, int max_iters, int boundary, const std::shared_ptr<CameraPyramid>& left_cam_ptr,
  //          const std::shared_ptr<CameraPyramid>& right_cam_ptr, float baseline, int max_residuals);
  //  GlobalStatus ComputeDepth(const cv::Mat& left_img, const cv::Mat& right_img, cv::Mat& left_val, cv::Mat& left_disp, cv::Mat& left_dep);

  odometry::GlobalStatus depth_state;
  // float search_min = (min_depth - 0.5f < 0.01f)? 0.01f : min_depth - 0.5f;
  // float search_max = (max_depth + 0.5f > 10.0f)? 10.0f : max_depth + 0.5f;
  float search_min = 3.0f;
  float search_max = 17.0f;
  std::cout << "constraint depth estimate range: " << search_min << " ~ " << search_max << std::endl;
  std::shared_ptr<odometry::CameraPyramid> left_cam_ptr = nullptr;
  std::shared_ptr<odometry::CameraPyramid> right_cam_ptr = nullptr;
  int max_residuals = 5000;
  odometry::DepthEstimator depth_est(35.0f, 1000.0f, 10.0f, search_min, search_max, 0.01f, 28.0f, 0.995f, 100, 4,
                                    left_cam_ptr, right_cam_ptr, baseline, max_residuals);
  cv::Mat left_val(gray[0].rows, gray[0].cols, CV_8U, init_val);
  cv::Mat left_disp(gray[0].rows, gray[0].cols, PixelType);
  cv::Mat left_dep(gray[0].rows, gray[0].cols, PixelType);
  std::cout << "start disparity & depth estimation..." << std::endl;
  depth_state = depth_est.ComputeDepth(gray[0], gray[1], left_val, left_disp, left_dep);


  if (depth_state != -1){
    std::cout << "compute succeed." << std::endl;
    std::cout << "number of val depth: " << cv::sum(left_val)[0] << std::endl;
    std::cout << std::endl << "Report Statistics:" << std::endl;
    depth_est.ReportStatus();
    // report error
    report_disp_error(left_disp, gt_disp[0], left_val, gt_valid_map);
    report_depth_error(left_dep, gt_depth, left_val, gt_valid_map);
    cv::Mat gray_left;
    gray[0].convertTo(gray_left, cv::IMREAD_GRAYSCALE);
    for (int y=0; y<left_val.rows; y++){
      for (int x=0; x<left_val.cols; x++){
        if (left_val.at<uint8_t>(y,x)==1 && gt_disp[0].at<float>(y,x) != 0){
          cv::circle(gray_left, cv::Point(x,y), 4, cv::Scalar(0));
        }
      }
    }
    cv::namedWindow("keypoints", cv::WINDOW_NORMAL);
    cv::imshow("keypoints", gray_left);
    cv::waitKey(0);
  } else {
    std::cout << "compute failed." << std::endl;
  }

  return 0;
}

void load_data(const std::string& folder_name, std::vector<cv::Mat> &gray, std::vector<cv::Mat> &disp){
  std::string left_img_file = folder_name + "/view1.png";
  std::string right_img_file = folder_name + "/view5.png";
  std::string left_disp_file = folder_name + "/disp1.png";
  std::string right_disp_file = folder_name + "/disp5.png";
  cv::Mat gray_8u;

  // read left image
  gray_8u = cv::imread(left_img_file, cv::IMREAD_GRAYSCALE);
  if (gray_8u.empty()){
    std::cout << "read left img failed." << std::endl;
    std::exit(-1);
  }
  //cv::imshow("left_img", gray_8u);
  gray_8u.convertTo(gray[0], PixelType);
  //std::cout << "left size: " << gray[0].size << std::endl;

  // read right image
  gray_8u = cv::imread(right_img_file, cv::IMREAD_GRAYSCALE);
  if (gray_8u.empty()){
    std::cout << "read right img failed." << std::endl;
    std::exit(-1);
  }
  //cv::imshow("right_img", gray_8u);
  gray_8u.convertTo(gray[1], PixelType);
  //std::cout << "right size: " << gray[1].size << std::endl;

  // read left disp
  gray_8u = cv::imread(left_disp_file, cv::IMREAD_GRAYSCALE);
  if (gray_8u.empty()){
    std::cout << "read left disp failed." << std::endl;
    std::exit(-1);
  }
  //cv::imshow("left_disp", gray_8u);
  gray_8u.convertTo(disp[0], PixelType, 1.0f);
  //std::cout << "left disp: " << disp[0].size << std::endl;

  // read right disp
  gray_8u = cv::imread(right_disp_file, cv::IMREAD_GRAYSCALE);
  if (gray_8u.empty()){
    std::cout << "read left disp failed." << std::endl;
    std::exit(-1);
  }
  //cv::imshow("right_disp", gray_8u);
  gray_8u.convertTo(disp[1], PixelType, 1.0f);
  //std::cout << "right disp: " << disp[1].size << std::endl;
  //std::cout << "rows: " << disp[1].rows << std::endl;
  //cv::waitKey(0);
}

void report_disp_error(const cv::Mat& pred_disp, const cv::Mat& gt_disp, const cv::Mat& valid_map, const cv::Mat& gt_valid_map){
  std::vector<int> statistic{0,0,0,0,0,0,0,0,0,0,0};
  std::vector<int> accmulate_stat{0,0,0,0,0,0,0,0,0,0,0};
  float err_sum = 0;
  float abs_err = 0;
  float sum_val = float(cv::sum(valid_map.mul(gt_valid_map))[0]);
  std::cout << "number of valid for comparing against gt disparity: " << sum_val << std::endl;
  for (int y=0; y<1110; y++){
    for (int x=0; x<1330; x++){
      if (valid_map.at<uint8_t>(y,x) == 1 && gt_valid_map.at<uint8_t>(y,x) == 1){
        abs_err = std::abs(gt_disp.at<float>(y,x) - pred_disp.at<float>(y,x));
        if (abs_err <= 0.5) statistic[0] += 1;
        else if (abs_err <= 1.0 && abs_err > 0.5) statistic[1] += 1;
        else if (abs_err <= 2.0 && abs_err > 1.0) statistic[2] += 1;
        else if (abs_err <= 3.0 && abs_err > 2.0) statistic[3] += 1;
        else if (abs_err <= 4.0 && abs_err > 3.0) statistic[4] += 1;
        else if (abs_err <= 5.0 && abs_err > 4.0) statistic[5] += 1;
        else if (abs_err <= 6.0 && abs_err > 5.0) statistic[6] += 1;
        else if (abs_err <= 7.0 && abs_err > 6.0) statistic[7] += 1;
        else if (abs_err <= 10.0 && abs_err > 7.0) statistic[8] += 1;
        else if (abs_err <= 20.0 && abs_err > 10) statistic[9] += 1;
        else statistic[10] += 1;
        err_sum += abs_err;
      } else
        continue;
    }
  }
  accmulate_stat[0] = statistic[0]; // pixel error <= 0.5
  for (int i=1; i<11; i++){
    accmulate_stat[i] = statistic[i] + accmulate_stat[i-1];
  }
  std::cout << "Disparity errors in [pixels]:" << std::endl;
  std::cout << "Error <= 0.5 : " << accmulate_stat[0] << ", percentage: " << float(accmulate_stat[0])/sum_val*100.0f << std::endl;
  std::cout << "Error <= 1 : " << accmulate_stat[1] << ", percentage: " << float(accmulate_stat[1])/sum_val*100.0f << std::endl;
  std::cout << "Error <= 2 : " << accmulate_stat[2] << ", percentage: " << float(accmulate_stat[2])/sum_val*100.0f << std::endl;
  std::cout << "Error <= 3 : " << accmulate_stat[3] << ", percentage: " << float(accmulate_stat[3])/sum_val*100.0f << std::endl;
  std::cout << "Error <= 4 : " << accmulate_stat[4] << ", percentage: " << float(accmulate_stat[4])/sum_val*100.0f << std::endl;
  std::cout << "Error <= 5 : " << accmulate_stat[5] << ", percentage: " << float(accmulate_stat[5])/sum_val*100.0f << std::endl;
  std::cout << "Error <= 6 : " << accmulate_stat[6] << ", percentage: " << float(accmulate_stat[6])/sum_val*100.0f << std::endl;
  std::cout << "Error <= 7 : " << accmulate_stat[7] << ", percentage: " << float(accmulate_stat[7])/sum_val*100.0f << std::endl;
  std::cout << "Error <= 10 : " << accmulate_stat[8] << ", percentage: " << float(accmulate_stat[8])/sum_val*100.0f << std::endl;
  std::cout << "Error <= 20 : " << accmulate_stat[9] << ", percentage: " << float(accmulate_stat[9])/sum_val*100.0f << std::endl;
  std::cout << "Error > 20 : " << statistic[9] << ", percentage: " << float(statistic[9])/sum_val*100.0f << std::endl;
  std::cout << "average disparity error [pixels]: " << err_sum / sum_val << std::endl;
}

void report_depth_error(const cv::Mat& pred_depth, const cv::Mat& gt_depth, const cv::Mat& valid_map, const cv::Mat& gt_valid_map){
  std::vector<int> statistic{0,0,0,0,0,0,0,0,0,0,0};
  std::vector<int> accmulate_stat{0,0,0,0,0,0,0,0,0,0,0};
  float sum_val = float(cv::sum(valid_map.mul(gt_valid_map))[0]);
  float abs_err; // in meters
  float err_sum = 0;
  for (int y=0; y<1110; y++){
    for (int x=0; x<1330; x++){
      if (valid_map.at<uint8_t>(y,x) == 1 && gt_valid_map.at<uint8_t>(y,x) == 1){
        abs_err = std::abs(1.0f/pred_depth.at<float>(y,x) - 1.0f/gt_depth.at<float>(y,x));
        if (abs_err <= 0.01) statistic[0] += 1;
        else if (abs_err <= 0.02 && abs_err > 0.01) statistic[1] += 1;
        else if (abs_err <= 0.03 && abs_err > 0.02) statistic[2] += 1;
        else if (abs_err <= 0.04 && abs_err > 0.03) statistic[3] += 1;
        else if (abs_err <= 0.05 && abs_err > 0.04) statistic[4] += 1;
        else if (abs_err <= 0.06 && abs_err > 0.05) statistic[5] += 1;
        else if (abs_err <= 0.07 && abs_err > 0.06) statistic[6] += 1;
        else if (abs_err <= 0.08 && abs_err > 0.07) statistic[7] += 1;
        else if (abs_err <= 0.1 && abs_err > 0.08) statistic[8] += 1;
        else if (abs_err <= 0.5 && abs_err > 0.1) statistic[9] += 1;
        else statistic[10] += 1;
        err_sum += abs_err;
      } else
        continue;
    }
  }
  accmulate_stat[0] = statistic[0]; // pixel error <= 0.01
  for (int i=1; i<11; i++){
    accmulate_stat[i] = statistic[i] + accmulate_stat[i-1];
  }
  std::cout << "Depth errors in [meters]:" << std::endl;
  std::cout << "Error <= 0.01 : " << accmulate_stat[0] << ", percentage: " << float(accmulate_stat[0])/sum_val*100.0f << std::endl;
  std::cout << "Error <= 0.02 : " << accmulate_stat[1] << ", percentage: " << float(accmulate_stat[1])/sum_val*100.0f << std::endl;
  std::cout << "Error <= 0.03 : " << accmulate_stat[2] << ", percentage: " << float(accmulate_stat[2])/sum_val*100.0f << std::endl;
  std::cout << "Error <= 0.04 : " << accmulate_stat[3] << ", percentage: " << float(accmulate_stat[3])/sum_val*100.0f << std::endl;
  std::cout << "Error <= 0.05 : " << accmulate_stat[4] << ", percentage: " << float(accmulate_stat[4])/sum_val*100.0f << std::endl;
  std::cout << "Error <= 0.06 : " << accmulate_stat[5] << ", percentage: " << float(accmulate_stat[5])/sum_val*100.0f << std::endl;
  std::cout << "Error <= 0.07 : " << accmulate_stat[6] << ", percentage: " << float(accmulate_stat[6])/sum_val*100.0f << std::endl;
  std::cout << "Error <= 0.08 : " << accmulate_stat[7] << ", percentage: " << float(accmulate_stat[7])/sum_val*100.0f << std::endl;
  std::cout << "Error <= 0.1 : " << accmulate_stat[8] << ", percentage: " << float(accmulate_stat[8])/sum_val*100.0f << std::endl;
  std::cout << "Error <= 0.5 : " << accmulate_stat[9] << ", percentage: " << float(accmulate_stat[9])/sum_val*100.0f << std::endl;
  std::cout << "Error > 0.5 : " << statistic[9] << ", percentage: " << float(statistic[9])/sum_val*100.0f << std::endl;
  std::cout << "average depth error [meters]: " << err_sum / sum_val << std::endl;
}