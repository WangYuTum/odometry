// The file contains (stereo) camera configuration implementations
// Created by Yu Wang on 2019-01-17.

#include <camera.h>
#include <fstream>
#include <stdlib.h>
#include <string>

namespace odometry
{

CameraPyramid::CameraPyramid(int levels, double fx, double fy, double f_theta, double cx, double cy,
                             int resolution_width, int resolution_height){
  resolution_width_ = resolution_width;
  resolution_height_ = resolution_height;
  levels_ = levels;
  // build pyramid
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
}

} // namespace odometry

