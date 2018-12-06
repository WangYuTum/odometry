// Created by Yu Wang on 03.12.18.
// The header file of Image Pyramid contains a pyramid of some grayscale/depth image.

#ifndef RGBD_ODOMETRY_IMAGE_PYRAMID_H
#define RGBD_ODOMETRY_IMAGE_PYRAMID_H

#include <opencv2/core.hpp>
#include <vector>
#include <data_types.h>

namespace odometry
{

class ImagePyramid{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // disable default constructor explicitly
    ImagePyramid() = delete;

    // parameterized constructor
    ImagePyramid(int num_levels, const cv::Mat& in_img);

    // disable copy constructor for now
    ImagePyramid(const ImagePyramid& ) = delete;

    // disable copy assignment for now
    ImagePyramid& operator= (const ImagePyramid& ) = delete;

    // get total number of pyramid levels, by default inline
    int GetNumberLevels() const{return num_levels_;};

    // get the image from the corresponding pyramid level, return as const reference
    const cv::Mat& GetPyramidImage(int level_idx) const;

  private:
    int num_levels_; // total number of pyramid levels, default to 4
    std::vector<cv::Mat> pyramid_imgs_; // vector of pyramid images, store the actual data, not pointers/reference
};

class DepthPyramid{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // disable default constructor explicitly
    DepthPyramid() = delete;

    // parameterized constructor
    DepthPyramid(int num_levels, const cv::Mat& in_depth);

    // disable copy constructor for now
    DepthPyramid(const DepthPyramid& ) = delete;

    // disable copy assignment for now
    DepthPyramid& operator= (const DepthPyramid& ) = delete;

    // get total number of pyramid levels
    int GetNumberLevels() const{return num_levels_;};

    // get the image from the corresponding pyramid level, return as reference
    const cv::Mat& GetPyramidDepth(int level_idx) const;

  private:
    int num_levels_; // total number of pyramid levels, default to 4
    std::vector<cv::Mat> pyramid_depths_; // vector of pyramid images, store the actual data, not pointers/reference

};

} // namespace odometry


#endif //RGBD_ODOMETRY_IMAGE_PYRAMID_H
