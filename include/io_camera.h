// This is an I/O wrapper for camera input, running as an independent thread (constantly reading camera)
// Created by Yu Wang on 2019-01-29.

#ifndef ODOMETRY_IO_CAMERA_H
#define ODOMETRY_IO_CAMERA_H

#include <iostream>
#include <camera.h>
#include "data_types.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

namespace odometry
{

void RunCamera(cv::Mat* current_left, cv::Mat* current_right, std::shared_ptr<odometry::CameraPyramid> cam_ptr_left,
               std::shared_ptr<odometry::CameraPyramid> cam_ptr_right, int cam_deviceID=0);


}



#endif //ODOMETRY_IO_CAMERA_H
