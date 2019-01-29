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
    // level: how many level you want to build
    // fx: focal_length [mm] * pixels_along_x/mm
    // fy: focal_length [mm] * pixels_along_y/mm
    // cx: principle point_x [pixels]
    // cy: principle point_y [pixels]
    // k1: radial distortion k1
    // k2: radial distortion k2
    // r1: tangential distortion r1
    // r2: tangential distortion r2
    // sensor_width: camera sensor width in [mm]
    // sensor_height: camera sensor height in [mm]
    // resolution_width: camera raw resolution width in [pixels]
    // resolution_width: camera raw resolution height in [pixels]
    CameraPyramid(int levels, double fx, double fy, double f_theta, double cx, double cy, double k1, double k2, double r1, double r2,
                  double sensor_width, double sensor_height, int resolution_width, int resolution_height);

    // disable copy constructor
    CameraPyramid(const CameraPyramid&) = delete;

    // disable copy assignment
    CameraPyramid& operator = (const CameraPyramid&) = delete;

    // MUST be called after stereoRectify, this function mainly computes remap for undistort & rectification
    // INPUT:
    //  * rectification rotation (computed from stereoRectify), used for generate remap
    //  * rectified new camera matrix 3x4 (computed from stereoRectify), assuming units are in [pixels]
    //  * size of undistorted & rectified image
    //  * remap type (opencv default: CV_32FC1)
    //  * whether convert floating-point remap to fixed-point remap for speed or not (default: false)
    // OUTPUT:
    //  * set internal new camera matrix and build pyramid
    //  * set internal remap for online undistort & remap camera images
    //            later we can use the remaps to undistort & rectify the raw camera inputs
    void ConfigureCamera(const cv::Mat& rectify_rotation, const cv::Mat& new_intrinsic,
            const cv::Size& new_size, int map_type, bool use_int_map);

    // MUST be called to undistort and rectify new raw camera inputs
    // Inputs:
    //  * src_raw: the camera raw output (MUST be width=640, height=480), a constant memory block that keeps receiving new camera frames
    //  * dst: MUST be dynamically allocated Matrix, which are used by depth estimate, camera tracking;
    //          if considered as a non-keyframe after tracking, the Matrix MUST be deleted.
    //  * other parameters have default values as OpenCV.
    // Outputs:
    //  * -1 if failed; otherwise success
    GlobalStatus UndistortRectify(const cv::Mat& src_raw, cv::Mat& dst, int interpolation=cv::INTER_LINEAR,
            int borderMode=cv::BORDER_CONSTANT, const cv::Scalar& borderValue=cv::Scalar());

    /*************** Accessor for rectified camera intrinsics ****************/
    float fx_float(int level){ return float(intrinsic_[level].at<double>(0, 0)); }  // unit: pixels, = fy
    float fy_float(int level){ return float(intrinsic_[level].at<double>(1, 1)); }  // unit: pixels, = fx
    float f_theta_float(int level){ return float(intrinsic_[level].at<double>(0, 1)); } // unit: pixels
    float cx_float(int level) { return float(intrinsic_[level].at<double>(0, 2)); } // unit: pixels
    float cy_float(int level) { return float(intrinsic_[level].at<double>(1, 2)); } // unit: pixels
    float f_meters_float(int level) {return float(intrinsic_[level].at<double>(0,0) / pixels_per_mm_x_);}

    double fx_double(int level){ return intrinsic_[level].at<double>(0, 0); }  // unit: pixels, = fy
    double fy_double(int level){ return intrinsic_[level].at<double>(1, 1); }  // unit: pixels, = fx
    double f_theta_double(int level){ return intrinsic_[level].at<double>(0, 1); } // unit: pixels
    double cx_double(int level) { return intrinsic_[level].at<double>(0, 2); } // unit: pixels
    double cy_double(int level) { return intrinsic_[level].at<double>(1, 2); } // unit: pixels
    double f_meters_double(int level) {return intrinsic_[level].at<double>(0,0) / pixels_per_mm_x_;}

    const cv::Mat& get_intrinsic_rectified(int level) { return intrinsic_[level]; }


    /********************* Accessor for hardware specs & raw camera parameters **********************/
    double sensor_w_double() { return sensor_width_; }
    double sensor_h_double() { return sensor_height_; }
    double pixels_per_mm_x_double() { return pixels_per_mm_x_; }
    double pixels_per_mm_y_double() { return pixels_per_mm_y_; }
    int resolution_raw_w() { return resolution_width_; }
    int resolution_raw_h() { return resolution_height_; }

    float sensor_w_float() { return float(sensor_width_); }
    float sensor_h_float() { return float(sensor_height_); }
    float pixels_per_mm_x_float() { return float(pixels_per_mm_x_); }
    float pixels_per_mm_y_float() { return float(pixels_per_mm_y_); }

    const cv::Mat& get_intrinsic_raw() { return intrinsic_raw_; }
    const cv::Mat& get_distortion_coeff() { return distortion_param_; }


  private:
    // we use cv::Mat since most camera related configurations are in opencv
    cv::Mat intrinsic_raw_;
    cv::Mat distortion_param_;
    int resolution_width_, resolution_height_;
    double sensor_width_, sensor_height_; // units are in [mm]
    double pixels_per_mm_x_, pixels_per_mm_y_;

    /****************************** parameters after camera configuration *********************************/
    int levels_;
    // The rectified camera intrinsic pyramid
    std::vector<cv::Mat> intrinsic_;
    // The remap params that will be use for undistort & rectify raw camera input
    cv::Mat rmap_[2]; // float-32
}; // class CameraPyramid

/******************************************** GLOBAL STEREO CAMERA SETUP ***********************************************/
// The stereo camera setup utilities, must be done during the system initialisation
// Inputs:
//  * cam_ptr_left: left camera shared pointer, do not need to be initialised
//  * cam_ptr_right: right camera shared pointer, do not need to be initialised
//  * valid_region: the valid image region after undistort and rectify, do not need to be initialised
GlobalStatus SetUpStereoCameraSystem(const std::string& stereo_file, int levels, std::shared_ptr<CameraPyramid>& cam_ptr_left,
                                     std::shared_ptr<CameraPyramid>& cam_ptr_right, cv::Rect& valid_region, double& baseline);
// Read Stereo Calibration file to get calibrated camera parameters
void ReadStereoCalibrationFile(const std::string& stereo_file, cv::Mat& camera_intrin_left, cv::Mat& camera_intrin_right,
                               cv::Mat& dist_left, cv::Mat& dist_right, cv::Mat& rotate_left_right, cv::Mat& translate_left_right,
                               double& sensor_w_left, double& sensor_h_left, double& sensor_w_right,
                               double& sensor_h_right, int& resolution_w, int& resolution_h);

} // namespace odometry

#endif //ODOMETRY_CAMERA_H
