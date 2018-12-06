# Direct Stereo/Monocular (Semi-)Dense Visual Odometry and 3D Reconstruction
This is a course project from 3d scanning and motion capture at [Technical University München](https://www.tum.de/en/).
The project implements a direct (semi-)dense image alignment for tracking a Stereo or Monocular Camera 
as the frontend and implement a Multi-view Stereo Reconstruction as the backend. This is only an re-implementation of
existing algorithms.

The project is still under **developing**.
 
### Related/Referenced Papers
* **Robust Odometry Estimation for RGB-D Cameras**, *C. Kerl, J. Sturm, D. Cremers*, In Proc. of the IEEE Int. Conf. on Robotics and Automation (ICRA), 2013
* **Dense Visual SLAM for RGB-D Cameras**, *C. Kerl, J. Sturm, D. Cremers*, In Proc. of the Int. Conf. on Intelligent Robot Systems (IROS), 2013.
* **LSD-SLAM: Large-Scale Direct Monocular SLAM**, *J. Engel, T. Schöps, D. Cremers*, ECCV '14
* **Semi-Dense Visual Odometry for a Monocular Camera**, *J. Engel, J. Sturm, D. Cremers*, ICCV '13

### Requirements and Dependencies

* CMake >= 3.8
* Clang >= 7.0 (or any other compilers like GCC or Intel, although not tested)
* C++ 14 standard (mainly for security reasons, especially for pointers)
* C++ standard libraries, Eigen >= 3.3, OpenCV >= 3.4, [Sophus](https://github.com/strasdat/Sophus)(camera motions as Lie Group and Lie Algebra, only needed for estimating camera pose and scene/map reconstruction)

### Build and Compile conventions

* **Optional** build every .cpp source file as **static** libraries to lib directory
* **Always** seperate .cpp and .h files into different folders
* **Always** put third_party libraries or codes into third_party directory
* **DO NOT** git commit build (including executables) related files such as *.a, *.o, etc. Instead you should put them into corresponding folders and ignore them in your .gitignore file
* **EXPLICITLY** enable SIMD vectorization and CPU arch optimization when compiling
* You may have a look at the file structures and coding style of this repository for more details

### Code Style and Conventions

We follow [Google C++ Style](https://google.github.io/styleguide/cppguide.html) in general.

#### Datatype Rules
* **AVOID** use raw data types, such as arrays, raw pointers. Instead use the following:
    * **Eigen::Array** for (large) multi-dimentional data types
    * **std::vector** for (small-medium) sequence data types
    * use **reference** for passing/returning function arguments, **always** use **smart pointers** instead of raw pointers
    * **smart pointers:**
        *  std::unique_ptr, std::make_unique
        *  std::shared_ptr, std::make_shared
    * **DO NOT** use **auto** pointers because it confuses other people reading your code
    * use raw pointers **VERY CAREFULLY** if must
* **USE** Eigen::Matrix **only** for linear algebra related storage and operations
* **USE** OpenCV cv::Mat **only** for image related data storage and operations
* **USE** 32-bit float for all floating point data, use 64-bit double if must
* **USE** 32-bit int for all integer data, **DO NOT** use **unsigned_int** because it is already been 
proved as a design mistake in C++ standard
* **Aliasing** in Eigen: be aware if you have the same Eigen object on both side of the expression.
* **Alignment** in Eigen if you have **fixed-size vectorizable object** only.


#### Performance Concerns
* **USE** fix-sized Eigen::Matrix or Eigen::Vector **only** for small matrices (total number of elements up to 16)
    * **Optional** make sure the number of elements is dividable by 4, if not padding with additional zeros
* **USE** dynamic-sized Eigen::Matrix or Eigen::Vector for **all** medium-large matrices (Eigen do memory alignment automatically for large dynamic matrices)


### TODOs

* cv::Mat type CV_32F range
* declare member function as const at the end or begin


# License
The source code is licensed under the GNU General Public License Version 3 (GPLv3), see http://www.gnu.org/licenses/gpl.html.




