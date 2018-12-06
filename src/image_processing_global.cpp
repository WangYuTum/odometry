// Created by Yu Wang on 04.12.18.
// This is an implementation of functions defined in ODOMETRY_IMAGE_PROCESSING_GLOBAL_H

#include <image_processing_global.h>


namespace odometry
{

// TODO, using native c++ for loop combined with openmp to warp entire image
void WarpImageNative(const cv::Mat& img_in, const Matrix44f& kTransMat, cv::Mat& warped_img){

}

// TODO, using sse to warp entire image
void WarpImageSse(const cv::Mat& img_in, const Matrix44f& kTransMat, cv::Mat& warped_img){

}



} // namespace odometry

