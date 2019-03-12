// Created by Yu Wang on 06.12.18.
// The header file contains camera calibration related classes/functions

#ifndef ODOMETRY_CAMERA_H
#define ODOMETRY_CAMERA_H

#include <data_types.h>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/calib3d.hpp>

namespace odometry
{

class CameraPyramid{
  public:

    // disable default constructor
    CameraPyramid() = delete;

    // parametrized constructor: raw camera parameters
    // level: how many levels you want to build
    // fx: focal_length [mm] * pixels_along_x/mm
    // fy: focal_length [mm] * pixels_along_y/mm
    // cx: principle point_x [pixels]
    // cy: principle point_y [pixels]
    // resolution_width: camera output resolution width in [pixels]
    // resolution_width: camera output resolution height in [pixels]
    CameraPyramid(int levels, double fx, double fy, double f_theta, double cx, double cy, int resolution_width, int resolution_height);

    // disable copy constructor
    CameraPyramid(const CameraPyramid&) = delete;

    // disable copy assignment
    CameraPyramid& operator = (const CameraPyramid&) = delete;

    /*************** Accessor for camera intrinsics ****************/
    float fx_float(int level){ return float(intrinsic_[level].at<double>(0, 0)); }  // unit: pixels
    float fy_float(int level){ return float(intrinsic_[level].at<double>(1, 1)); }  // unit: pixels
    float f_theta_float(int level){ return float(intrinsic_[level].at<double>(0, 1)); } // unit: pixels
    float cx_float(int level) { return float(intrinsic_[level].at<double>(0, 2)); } // unit: pixels
    float cy_float(int level) { return float(intrinsic_[level].at<double>(1, 2)); } // unit: pixels

    double fx_double(int level){ return intrinsic_[level].at<double>(0, 0); }  // unit: pixels
    double fy_double(int level){ return intrinsic_[level].at<double>(1, 1); }  // unit: pixels
    double f_theta_double(int level){ return intrinsic_[level].at<double>(0, 1); } // unit: pixels
    double cx_double(int level) { return intrinsic_[level].at<double>(0, 2); } // unit: pixels
    double cy_double(int level) { return intrinsic_[level].at<double>(1, 2); } // unit: pixels


    /********************* Accessor for hardware specs & raw camera parameters **********************/
    int resolution_raw_w() { return resolution_width_; }
    int resolution_raw_h() { return resolution_height_; }

  private:
    // we use cv::Mat since most camera related configurations are in opencv
    int resolution_width_, resolution_height_;
    int levels_;
    std::vector<cv::Mat> intrinsic_;
}; // class CameraPyramid

} // namespace odometry

#endif //ODOMETRY_CAMERA_H
