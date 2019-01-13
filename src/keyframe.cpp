// The file contains declearation of keyframe. Whenever a keyframe is selected, a corresponding object is created.
// Created by Yu Wang on 2019-01-13.

#include <keyframe.h>

namespace odometry
{

KeyFrame::KeyFrame(const std::shared_ptr<cv::Mat>& kLeftImg, const std::shared_ptr<cv::Mat>& kRightImg,
         const std::shared_ptr<cv::Mat>& kLeftDep, const std::shared_ptr<cv::Mat>& kLeftVal, const Affine4f kAbsoPose){
  left_img_ptr_ = kLeftImg;
  right_img_ptr_ = kRightImg;
  left_dep_ptr_ = kLeftDep;
  left_val_ptr_ = kLeftVal;
  abso_pose_ = kAbsoPose;
}

KeyFrame::~ KeyFrame(){
  left_img_ptr_.reset();
  right_img_ptr_.reset();
  left_dep_ptr_.reset();
  left_val_ptr_.reset();
}

/****************************** Public const accessors ******************************/
const Affine4f KeyFrame::GetAbsoPose() { return abso_pose_; }
// The following accessors may cause run-time errors if the current keyframe is already deleted
const cv::Mat& KeyFrame::GetLeftImg(){ return *left_img_ptr_; }
const cv::Mat& KeyFrame::GetRightImg(){ return *right_img_ptr_; }
const cv::Mat& KeyFrame::GetLeftDep(){ return *left_dep_ptr_; }
const cv::Mat& KeyFrame::GetLeftVal(){ return *left_val_ptr_; }

/*********************** Public modifiers by returning non-const reference ***************************/
cv::Mat& KeyFrame::ModifyLeftDep() {return *left_dep_ptr_; }
cv::Mat& KeyFrame::ModifyLeftVal() {return *left_val_ptr_; }
Affine4f& KeyFrame::ModifyAbsoPose() {return abso_pose_; }

} // namespace odometry