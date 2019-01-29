// The file defines a frame that need to be shared between multiple threads
// Created by Yu Wang on 2019-01-29.

#include <frame.h>
#include <data_types.h>

namespace odometry
{

Frame::Frame(int rows, int cols){
  if (rows != 480 || cols != 640){
    std::cout << "ERROR! Creating Frame: rows != 480 || cols != 640" << std::endl;
  }
  img_mat_.create(rows, cols, PixelType);
}

Frame::Frame(const cv::Mat& src){
  if (src.rows != 480 || src.cols != 640){
    std::cout << "ERROR! Creating Frame: rows != 480 || cols != 640" << std::endl;
  }
  img_mat_ = src.clone();
}

Frame::Modify(const cv::Mat& src){
  img_mat_ = src.clone();
}

const cv::Mat& Frame::GetMat(){
  return img_mat_;
}

}

