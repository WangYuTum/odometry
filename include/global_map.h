// The file contains declaration of global_map which consists all keyframes
// Created by Yu Wang on 2019-01-13.

#ifndef ODOMETRY_GLOBAL_MAP_H
#define ODOMETRY_GLOBAL_MAP_H

#include <data_types.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <keyframe.h>
#include <iostream>
// #include <opencv2/highgui.hpp>
// #include <math.h>

namespace odometry
{

class GlobalMap{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /************************************ Necessary constructors/destructors **************************************/
    // explicit default constructor
    GlobalMap() = explicit;

    // explicit destructor, need reset shared_ptr to each keyframe
    ~ GlobalMap() = explicit;

    // disable copy constructor
    GlobalMap(const GlobalMap&) = delete;

    // disable copy assignment
    GlobalMap& operator= (const GlobalMap&) = delete;

    /************************************ Public const accessors **************************************/
    const KeyFrame& GetCurrentKeyFrame();
    int GetCurrentKeyFrameId();
    const KeyFrame& GetSingleKeyFrame(int key_index);
    const std::vector<std::shared_ptr<KeyFrame>>& GetKeyFrames();
    int GetNumKeyFrames();

    /*********************** Public modifiers by returning non-const reference ***************************/
    // No modifier for number of keyframes since it can only be modified through inserting new keyframes
    // the keyframe will be created via making a shared pointer, then passed to this method
    void InsertKeyFrame(std::shared_ptr<KeyFrame>& key_frame);
    KeyFrame& ModifySingleKeyFrame(int key_index);
    std::vector<std::shared_ptr<KeyFrame>>& ModifyKeyFrames();
    void SetCurrentKeyFrame();


  private:
    int num_key_frames_;
    std::vector<std::shared_ptr<KeyFrame>> key_frames_;
    int current_keyframe_id_; // -1 indicates invalid
};


} // namespace odometry

#endif //ODOMETRY_GLOBAL_MAP_H
