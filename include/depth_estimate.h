// The files contains all declaritions regarding to depth estimation
// Created by Yu Wang on 2019-01-11.

#ifndef ODOMETRY_DEPTH_ESTIMATE_H
#define ODOMETRY_DEPTH_ESTIMATE_H

#include <data_types.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>

namespace odometry
{

// NOTE that all input/output (or intermediate) images MUST be aligned against 32bit address
class DepthEstimator{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // disable default constructor explicitly
    DepthEstimator() = delete;

    // TODO: parameterized constructor, need camear parameters
    DepthEstimator();

    // disable copy constructor
    DepthEstimator(const DepthEstimator& ) = delete;

    // disable copy assignment
    DepthEstimator& operator= ( const DepthEstimator & ) = delete;

    // compute the depth of left image given a pair of stereo images
    // INPUT: pair of images MUST have been undistorted, and both images MUST be aligned to 32bit address
    // OUTPUT:
    //       * (Temporal for display)disparity map of (rectified) left image
    //       * depth map of left image
    //       * boolean valid map of left image (only the true pixels have valid depth, therefore be used for tracking)
    // Return: -1 if failed; otherwise success
    GlobalStatus ComputeDepth(const cv::Mat& left_img, const cv::Mat& right_img, cv::Mat& left_val, cv::Mat& left_disp, cv::Mat& left_dep);

  private:

    /************************************* Private data **************************************************/
    // Mainly camera parameters and (small) intermediate data


    /************************************** Methods used internally ********************************************/

    // rectify stereo images
    void RectifyStereo(const cv::Mat& left_img, const cv::Mat& right_img, cv::Mat& left_rect, cv::Mat& right_rect);

    // method that actually solve the disparity match and inverse depth estimation
    // Input:
    //    * rectified left img
    //    * rectified right img
    //    * grad of left (rectified) img
    // Output:
    //    * (Temporal for display) disparity map of rectified left img
    //    * depth map of left img
    //    * boolean valid map of left img
    void DisparityDepthEstimate(const cv::Mat& left_rect, const cv::Mat& right_rect, const cv::Mat& left_grad, cv::Mat& left_disp, cv::Mat& left_dep, cv::Mat& left_val);
};

}

#endif //ODOMETRY_DEPTH_ESTIMATE_H
