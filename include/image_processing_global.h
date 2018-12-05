// Created by Yu Wang on 04.12.18.
// The header file contains global function for different kinds of image processing, such as:
// image smoothing, image gradient, image pyramids, warping, etc.

// The functions can be used by all other classes and functions, unless you need to
// access private data members or member functions from some class, you need to explicitly declare the function as
// a global friend function within the class although it is highly unrecommended.

#ifndef ODOMETRY_IMAGE_PROCESSING_GLOBAL_H
#define ODOMETRY_IMAGE_PROCESSING_GLOBAL_H

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <data_types.h>

namespace odometry
{


void WarpImage(const cv::Mat& img_in, const Vector6f& twist, cv::Mat& warped_img); // TODO

// compute gaussian pyramid and save the value, the out_pyramids is not initialised
// return status: -1 failed, otherwise success
GlobalStatus GaussianImagePyramid(int num_levels, const cv::Mat& in_img, std::vector<cv::Mat>& out_pyramids); // TODO

} // namespace odometry

#endif //ODOMETRY_IMAGE_PROCESSING_GLOBAL_H
