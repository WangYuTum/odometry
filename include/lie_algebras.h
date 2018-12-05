// Created by Yu Wang on 05.12.18.
// The header file is a wrapper of Sophus(C++ implementation of LieGroup & LieAlgebra using Eigen). Since we won't need all functions
// from Sophus(like derivatives w.r.t SE(3) or se(3)), we just wrap the necessary classes/functions here for this project.

#ifndef ODOMETRY_LIE_ALGEBRAS_H
#define ODOMETRY_LIE_ALGEBRAS_H

#include <Eigen/Core>
#include <data_types.h>
#include <se3.hpp>


namespace odometry
{



} // namespace odometry

#endif //ODOMETRY_LIE_ALGEBRAS_H
