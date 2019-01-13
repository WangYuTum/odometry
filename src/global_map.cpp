// The file contains definition of global_map which consists all keyframes
// Created by Yu Wang on 2019-01-13.

#include <global_map.h>


namespace odometry
{

GlobalMap::GlobalMap(){
  num_key_frames_ = 0;
  current_keyframe_id_ = -1;
}

GlobalMap::~ GlobalMap(){
  for (int idx=0; idx<num_key_frames_; idx++){
    key_frames_[idx].reset();
  }
}

/************************************ Public const accessors **************************************/
const KeyFrame& GlobalMap::GetCurrentKeyFrame(){
  if (current_keyframe_id_ == -1){
    std::cout << "Current KeyFrame is not valid!" << std::endl;
    // TODO: terminate
  } else {
    return *key_frames_[current_keyframe_id_];
  }
}

int GlobalMap::GetCurrentKeyFrameId(){
  if (current_keyframe_id_ == -1)
    std::cout << "Current KeyFrame is not valid!" << std::endl;
  return current_keyframe_id_;
}

const KeyFrame& GlobalMap::GetSingleKeyFrame(int key_index){
  if (key_index<0 || key_index>=num_key_frames_){
    std::cout << "Invalid KeyFrame id!" << std::endl;
    // TODO: terminate
  } else {
    return *key_frames_[key_index];
  }
}

const std::vector<std::shared_ptr<KeyFrame>>& GlobalMap::GetKeyFrames(){
  return key_frames_;
}

int GlobalMap::GetNumKeyFrames(){
  return num_key_frames_;
}

/*********************** Public modifiers by returning non-const reference ***************************/
void GlobalMap::InsertKeyFrame(std::shared_ptr<KeyFrame>& key_frame){
  key_frames_.push_back(key_frame);
  num_key_frames_ += 1;
  current_keyframe_id_ += 1; // automatically change the current keyframe id
  std::cout << "Insert new KeyFrame. Current KeyFrame: " << current_keyframe_id_ << ". Total now: " << num_key_frames_ << "." << std::endl;
}

KeyFrame& GlobalMap::ModifySingleKeyFrame(int key_index){
  if (key_index<0 || key_index>=num_key_frames_){
    std::cout << "Invalid KeyFrame id!" << std::endl;
    // TODO: terminate
  } else {
    return *key_frames_[key_index];
  }
}

std::vector<std::shared_ptr<KeyFrame>>& GlobalMap::ModifyKeyFrames(){
  return key_frames_;
}

// change key frame id manually, rarely used; the current keyframe id is automatically changed after inserting a new keyframe
void GlobalMap::SetCurrentKeyFrame(int id){
  if (id<0 || id>num_key_frames_){
    std::cout << "Invalid KeyFrame id! Change KeyFrame Id failed! Current id: " << current_keyframe_id_ << std::endl;
  } else {
    std::cout << "Changed current KeyFrame from " << current_keyframe_id_ << " to " << id << std::endl;
    current_keyframe_id_ = id;
  }
}

} // namespace odometry