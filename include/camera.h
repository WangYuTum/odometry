// Created by Yu Wang on 06.12.18.
// The header file contains camera calibration related classes/functions

#ifndef ODOMETRY_CAMERA_H
#define ODOMETRY_CAMERA_H

#include <data_types.h>

namespace odometry
{

class CameraPyramid{
  public:

    // disable default constructor
    CameraPyramid() = delete;

    // parametrized constructor
    CameraPyramid(int levels, float fx, float fy, float f_theta, float cx, float cy){
      levels_ = levels;
      for (int l = 0; l < levels; l++){
        Matrix33f tmp;
        tmp(0, 0) = fx;
        tmp(1, 1) = fy;
        tmp(0, 2) = cx;
        tmp(1, 2) = cy;
        tmp(2, 2) = 1;
        tmp(0, 1) = f_theta;
        tmp(1, 0) = 0;
        tmp(2, 1) = 0;
        intrinsic_.emplace_back(tmp);
        fx = fx / 2.0f;
        fy = fy / 2.0f;
        f_theta = f_theta / 2.0f;
        cx = (cx + 0.5f) / 2.0f + 0.5f;
        cy = (cy + 0.5f) / 2.0f + 0.5f;
      }
    }

    // disable copy constructor
    CameraPyramid(const CameraPyramid&) = delete;

    // disable copy assignment
    CameraPyramid& operator = (const CameraPyramid&) = delete;

    float fx(int level){ return intrinsic_[level](0, 0); }
    float fy(int level){ return intrinsic_[level](1, 1); }
    float f_theta(int level){ return intrinsic_[level](0, 1); }
    float cx(int level) { return intrinsic_[level](0, 2); }
    float cy(int level) { return intrinsic_[level](1, 2); }

  private:

    int levels_;
    std::vector<Matrix33f> intrinsic_;
};

}

#endif //ODOMETRY_CAMERA_H
