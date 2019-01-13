// The files contains all definitions regarding to depth estimation
// Created by Yu Wang on 2019-01-11.

#include <depth_estimate.h>

namespace odometry
{

// TODO: parameter list
DepthEstimator::DepthEstimator(){

}

GlobalStatus DepthEstimator::ComputeDepth(const cv::Mat& left_img, const cv::Mat& right_img, cv::Mat& left_val,
        cv::Mat& left_disp, cv::Mat& left_dep){
  // Note that left_val, (rectified) left_disp, left_dep are already declared and aligned to 32bit address
  if ((left_img.rows != right_img.rows) || (left_img.cols != right_img.cols)){
    std::cout << "Number of rows/cols do not match for left/right images." << std::endl;
    return -1;
  }
  if ((left_img.type() != PixelType) || right_img.type() != PixelType){
    std::cout << "Pixel type of left/right images not 32-bit float." << std::endl;
    return -1;
  }
  if ((left_img.rows != 480) || left_img.cols != 640){
    std::cout << "rows != 480 or cols != 640." << std::endl;
    return -1;
  }

  // rectify left/right images
  cv::Mat left_rect(480, 640, PixelType);
  cv::Mat right_rect(480, 640, PixelType);
  RectifyStereo(left_img, right_img, left_rect, right_rect);

  // gradient of rectified left image
  // TODO: gradient

  // disparity search and depth estimation

}

void DepthEstimator::RectifyStereo(const cv::Mat& left_img, const cv::Mat& right_img, cv::Mat& left_rect, cv::Mat& right_rect){

}

void DepthEstimator::DisparityDepthEstimate(const cv::Mat& left_rect, const cv::Mat& right_rect, const cv::Mat& left_grad,
        cv::Mat& left_disp, cv::Mat& left_dep, cv::Mat& left_val){

}

}

