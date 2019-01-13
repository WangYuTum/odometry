// The file contains declaration of keyframe. Whenever a keyframe is selected, a corresponding object is created.
// Therefore, the keyframe object can only be created through dynamic memory allocator
// Created by Yu Wang on 2019-01-13.

#ifndef ODOMETRY_KEYFRAME_H
#define ODOMETRY_KEYFRAME_H

#include <data_types.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>
// #include <opencv2/highgui.hpp>
// #include <math.h>

namespace odometry
{

class KeyFrame {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /************************************ Necessary constructors/destructors **************************************/

    // disable default constructor explicitly
    KeyFrame() = delete;

    // parameterized constructor
    KeyFrame(const std::shared_ptr<cv::Mat>& kLeftImg, const std::shared_ptr<cv::Mat>& kRightImg,
            const std::shared_ptr<cv::Mat>& kLeftDep, const std::shared_ptr<cv::Mat>& kLeftVal, const Affine4f kAbsoPose);

    // destructor to release the pointer
    ~ KeyFrame();

    // disable copy constructor
    KeyFrame(const KeyFrame& ) = delete;

    // disable copy assignment
    KeyFrame& operator= ( const KeyFrame & ) = delete;

    /************************************ Public const accessors **************************************/
    const cv::Mat& GetLeftImg();
    const cv::Mat& GetRightImg();
    const cv::Mat& GetLeftDep();
    const cv::Mat& GetLeftVal();
    const Affine4f GetAbsoPose();

    /*********************** Public modifiers by returning non-const reference ***************************/
    // The depth values, depth valid mask, absolute pose might change during global optimization
    cv::Mat& ModifyLeftDep();
    cv::Mat& ModifyLeftVal();
    Affine4f& ModifyAbsoPose();

  private:

    /*** Shared Mem pointers, the corresponding memories will always be valid during the life time of odometry system ***/
    std::shared_ptr<cv::Mat> left_img_ptr_;
    std::shared_ptr<cv::Mat> right_img_ptr_;
    std::shared_ptr<cv::Mat> left_dep_ptr_;
    std::shared_ptr<cv::Mat> left_val_ptr_;
    Affine4f abso_pose_;
};

} // namespace odometry

#endif //ODOMETRY_KEYFRAME_H
