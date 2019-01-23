// The files contains all declaritions regarding to depth estimation.
// Created by Yu Wang on 2019-01-11.

#ifndef ODOMETRY_DEPTH_ESTIMATE_H
#define ODOMETRY_DEPTH_ESTIMATE_H

#include <data_types.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <math.h>
#include <iostream>
#include "camera.h"
#include <immintrin.h> // AVX instruction set
#include <pmmintrin.h> // SSE3
#include <xmmintrin.h> // SSE

namespace odometry
{

// NOTE that all input/output (or intermediate) images MUST be aligned against 32bit address
class DepthEstimator{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // disable default constructor explicitly
    DepthEstimator() = delete;

    DepthEstimator(float grad_th, float ssd_th, float photo_th, float min_depth, float max_depth, float lambda, float huber_delta,
            float precision, int max_iters, int boundary, const std::shared_ptr<CameraPyramid>& left_cam_ptr,
            const std::shared_ptr<CameraPyramid>& right_cam_ptr, float baseline, int max_residuals);

    // destructor, release shared_ptr of cameras
    ~DepthEstimator();

    // disable copy constructor
    DepthEstimator(const DepthEstimator& ) = delete;

    // disable copy assignment
    DepthEstimator& operator= ( const DepthEstimator & ) = delete;

    // compute the depth of left image given a pair of stereo images
    // INPUT: pair of images MUST have been UNDISTORTED and RECTIFIED, and both images MUST be aligned to 32bit address.
    // OUTPUT:
    //       * (Temporal for display)disparity map of left image
    //       * depth map of left image
    //       * 1/0 valid map of left image (only the true pixels have valid depth, therefore be used for tracking)
    // Return: -1 if failed; otherwise success
    GlobalStatus ComputeDepth(const cv::Mat& left_img, const cv::Mat& right_img, cv::Mat& left_val, cv::Mat& left_disp, cv::Mat& left_dep);

  private:

    /************************************* Private data **************************************************/
    // cameras: both intrinsics and extrinsics are needed for optimizing depth map
    std::shared_ptr<CameraPyramid> camera_ptr_left_;
    std::shared_ptr<CameraPyramid> camera_ptr_right_;
    float baseline_; // translation between left/right cameras in x-axis after rectification, unit in [meters]

    // left_right_translate_

    int boundary_;  // number of pixel ignored on the image boundary, determined by rectification and pre-defined, multiple of 4
    float grad_th_; // pixel gradient smaller than this will be ignored in disparity search
    float ssd_th_;  // ssd error over 8 neighbourhood pixels larger than this will be ignored in disparity search
    // optimizer related params
    float min_depth_; // depth values smaller than this will be ignored in disparity search & after optimization
    float max_depth_; // depth values larger than this will be ignored in disparity search & after optimization
    float photo_th_; // photometric error larger than this will be ignored as outliers after optimization
    int max_residuals_; // the max number of residuals allowed for depth optimization, default=10000
    float huber_delta_;
    float lambda_;  // do not change it
    float precision_;
    int max_iters_;
    int iters_stat_;  // reset to 0 before each call automatically
    float cost_stat_; // reset to 0 before each call automatically


    /************************************** Methods used internally ********************************************/

    // method that actually solve the disparity match and inverse depth estimation
    // Input:
    //    * rectified left img
    //    * rectified right img
    //    * grad of left (rectified) img
    // Output:
    //    * (Temporal for display) disparity map of rectified left img
    //    * depth map of left img
    //    * one/zero valid map of left img
    // Return:
    //    * -1 if failed
    GlobalStatus DisparityDepthEstimate(const cv::Mat& left_rect, const cv::Mat& right_rect, cv::Mat& left_disp, cv::Mat& left_dep, cv::Mat& left_val);

    // depth optimization after initial disparity search, no pyramid
    // Parameter list:
    //  * rectified left img
    //  * rectified right img
    //  * left depth map (changed after optimization)
    //  * left valid map (changed after optimization)
    GlobalStatus DepthOptimization(const cv::Mat& left_rect, const cv::Mat& right_rect, cv::Mat& left_dep, cv::Mat& left_val);
    GlobalStatus ComputeResidualJacobian(const Eigen::Matrix<float, Eigen::Dynamic, 1>& tmp_depth,
                                         const std::vector<std::pair<int, int>>& coord_vec,
                                         Eigen::DiagonalMatrix<float, Eigen::Dynamic>& jtwj, Eigen::Matrix<float, Eigen::Dynamic, 1>& b,
                                         float& err_now, const int num_residuals, const cv::Mat& left_img, const cv::Mat& right_img,
                                         Eigen::Matrix<float, Eigen::Dynamic, 1>& residuals);

    // compute ssd error 5x5 given all the image row pointers
    inline float ComputeSsd5x5(const float* left_pp_row_ptr, const float* left_p_row_ptr, const float* left_row_ptr, const float* left_n_row_ptr, const float* left_nn_row_ptr,
            const float* right_pp_row_ptr, const float* right_p_row_ptr, const float* right_row_ptr, const float* right_n_row_ptr, const float* right_nn_row_ptr,
            int left_x, int right_x);
    // compute ssd error using path pattern from DSO paper
    inline float ComputeSsdPattern8(const float* left_pp_row_ptr, const float* left_p_row_ptr, const float* left_row_ptr, const float* left_n_row_ptr, const float* left_nn_row_ptr,
                             const float* right_pp_row_ptr, const float* right_p_row_ptr, const float* right_row_ptr, const float* right_n_row_ptr, const float* right_nn_row_ptr,
                             int left_x, int right_x);
    // compute ssd error using path pattern from DSO paper, use sse impl.
    inline void ComputeSsdPattern8Sse(const __m256& left_pattern, const float* right_pp_row_ptr, const float* right_p_row_ptr,
                                                const float* right_row_ptr, const float* right_n_row_ptr, const float* right_nn_row_ptr, int x, float* result);
    // compute ssd error along one-dim epl
    inline float ComputeSsdLine(const float* left_row_ptr, const float* right_row_ptr, int left_x, int right_x);
};

} // namespace odometry

#endif //ODOMETRY_DEPTH_ESTIMATE_H
