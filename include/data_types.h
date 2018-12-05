// Created by Yu Wang on 03.12.18.
// The header file contains customized data types.

#ifndef RGBD_ODOMETRY_DATA_TYPES_H
#define RGBD_ODOMETRY_DATA_TYPES_H

#include <Eigen/Core>

typedef Eigen::Vector4i Vector4i;
typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

typedef int OptimizerStatus;

#endif //RGBD_ODOMETRY_DATA_TYPES_H

