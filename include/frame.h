// The file defines a frame that need to be shared between multiple threads
// Created by Yu Wang on 2019-01-29.

#ifndef ODOMETRY_FRAME_H
#define ODOMETRY_FRAME_H

#include <boost/thread.hpp>
#include <opencv2/core.hpp>
#include "data_types.h"

namespace odometry
{

class Frame{
  public:

    // constructor1: MUST specify size
    Frame(int rows, int cols);

    // constructor2
    Frame(const cv::Mat& src);

    // Modifier, deep copy
    Modify(const cv::Mat& src);

    // Accessor
    const cv::Mat& GetMat();

  private:

    // mutex to shared cv::Mat data
    boost::shared_mutex shared_mutex_;
    cv::Mat img_mat_;
};

}



#endif //ODOMETRY_FRAME_H
