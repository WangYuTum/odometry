CMake & Language & Libraries
* We use CMake >= 3.8
* We use Clang >= 7.0 (or any other compilers like GCC or Intel)
* We use C++14(for efficiency and security reasons, especially for pointers)
* We use C++ standard libs, Eigen >= 3.3, OpenCV >= 3.4, Sophus (LieGroup, LieAlgebra)


DataType rules:
* AVOID use raw data types, such as arrays, pointers. Instead use the following:
    * multi-dimentional(large) data types: Eigen::Array
    * sequence(small-medium) data types: std::vector
    * use reference instead of pointers if you can, use smart pointers always 
    * smart pointers: 
        * std::unique_ptr, std::make_unique
        * std::shared_ptr, std::make_shared
    * use raw pointers VERY CAREFULLY if you must
* USE Eigen Matrix only for linear algebra
* USE OpenCV Mat only for images(rgb, grayscale, depth)
* USE 32-bit float(default) for all floating point data, use 64-bit double if you have to
* USE 32-bit int(default) for all integer data


TODO checks:
* cv::Mat type CV_32F range
* declare member function as const at the end or begin

Eigen Matrix rules/optimizations:
* USE fix sized Matrix/Vector only for small matrices(total number of elements up to 16, make sure number of elements is 
dividable by 4)
* USE dynamic sized Matrix/Vector for all medium-large matrices(general reason: Eigen do memory alignment automatically for 
large dynamic matrices)
* EXPLICITLY enable SIMD vectorization and CPU arch optimization when compiling
