// Created by Yu Wang on 06.12.18.
// The header file contains camera calibration related classes/functions

#ifndef ODOMETRY_CAMERA_H
#define ODOMETRY_CAMERA_H

#include <data_types.h>

namespace odometry
{

class Camera{
  public:

    // disable default constructor
    Camera() = delete;

    // parametrized constructor
    Camera(float fx, float fy, float f_theta, float cx, float cy): fx_(fx), fy_(fy), f_theta_(f_theta), cx_(cx), cy_(cy){
      intrinsic_(0, 0) = fx_;
      intrinsic_(1, 1) = fy_;
      intrinsic_(0, 2) = cx_;
      intrinsic_(1, 2) = cy_;
      intrinsic_(2, 2) = 1;
      intrinsic_(0, 1) = f_theta_;
      intrinsic_(1, 0) = 0;
      intrinsic_(2, 1) = 0;
    }

    // disable copy constructor
    Camera(const Camera&) = delete;

    // disable copy assignment
    Camera& operator = (const Camera&) = delete;

    const float fx_, fy_, f_theta_, cx_, cy_;

    const Matrix33f& GetIntrisic(){
      return intrinsic_;
    }

  private:

    Matrix33f intrinsic_;
};

}

#endif //ODOMETRY_CAMERA_H
