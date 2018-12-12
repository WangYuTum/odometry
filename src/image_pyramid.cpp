// Created by Yu Wang on 03.12.18.
//

#include <image_pyramid.h>
#include <image_processing_global.h>
#include <iostream>

namespace odometry
{

//*************************** Image Pyramid *****************************//

ImagePyramid::ImagePyramid(int num_levels, const cv::Mat& in_img, bool smooth=true){
  num_levels_ = num_levels;
  GlobalStatus status = GaussianImagePyramidNaive(num_levels_, in_img, pyramid_imgs_, smooth);
  if (status == -1){
    std::cout << "Compute Gaussian Image Pyramid failed!" << std::endl;
  }
}

const cv::Mat& ImagePyramid::GetPyramidImage(int level_idx) const{
  if (level_idx >= num_levels_){
    std::cout << "Requested image pyramid does not exist! Max pyramid id: " << num_levels_ - 1 << std::endl;
    exit(1);
  }
  return pyramid_imgs_[level_idx];
}

//*************************** Depth Map Pyramid *****************************//
DepthPyramid::DepthPyramid(int num_levels, const cv::Mat& in_depth, bool smooth=true){
  num_levels_ = num_levels;
  GlobalStatus status = MedianDepthPyramidNaive(num_levels_, in_depth, pyramid_depths_, smooth);
  if (status == -1){
    std::cout << "Compute Gaussian Depth Pyramid failed!" << std::endl;
  }
}

const cv::Mat& DepthPyramid::GetPyramidDepth(int level_idx) const{
  return pyramid_depths_[level_idx];
}


} // namespace odometry
