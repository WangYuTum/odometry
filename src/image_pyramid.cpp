// Created by Yu Wang on 03.12.18.
//

#include <image_pyramid.h>
#include <image_processing_global.h>
#include <iostream>

namespace odometry
{

//*************************** Image Pyramid *****************************//

ImagePyramid::ImagePyramid(int num_levels, const cv::Mat& in_img){
  num_levels_ = num_levels;
  GlobalStatus status = GaussianImagePyramid(num_levels_, in_img, pyramid_imgs_);
  if (status == -1){
    std::cout << "Compute Gaussian Image Pyramid failed!" << std::endl;
  }
}

const cv::Mat& ImagePyramid::GetPyramidImage(int level_idx) const{
  return pyramid_imgs_[level_idx];
}

//*************************** Depth Map Pyramid *****************************//
DepthPyramid::DepthPyramid(int num_levels, const cv::Mat& in_depth){
  num_levels_ = num_levels;
  GlobalStatus status = GaussianImagePyramid(num_levels_, in_depth, pyramid_depths_);
  if (status == -1){
    std::cout << "Compute Gaussian Depth Pyramid failed!" << std::endl;
  }
}

const cv::Mat& DepthPyramid::GetPyramidDepth(int level_idx) const{
  return pyramid_depths_[level_idx];
}


} // namespace odometry
