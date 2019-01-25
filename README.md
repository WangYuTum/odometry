# Direct Stereo Semi-Dense Visual Odometry and 3D Reconstruction
This is a course project from 3d scanning and motion capture at [Technical University München](https://www.tum.de/en/).
The project implements a direct semi-dense image alignment for tracking a Stereo Camera 
as the frontend and implement a Multi-view Stereo Reconstruction as the backend. This is only an re-implementation and 
combination of existing algorithms. Some of the highlights:
* **Real-time:** on a single Intel CPU core (>= 4th generation)
* **Customized Optimization:** all optimization procedures are implemented from scratch, no [ceres-solver](http://ceres-solver.org/) or other optimization frameworks are used
* **Algorithm Dependencies:** all important algorithms are implemented from scratch, such as disparity search, depth/geometry optimization, pose optimization, etc

The project is still under **developing**.
 
### Related/Referenced Papers
* **Robust Odometry Estimation for RGB-D Cameras**, *C. Kerl, J. Sturm, D. Cremers*, In Proc. of the IEEE Int. Conf. on Robotics and Automation (ICRA), 2013.
* **Dense Visual SLAM for RGB-D Cameras**, *C. Kerl, J. Sturm, D. Cremers*, In Proc. of the Int. Conf. on Intelligent Robot Systems (IROS), 2013.
* **LSD-SLAM: Large-Scale Direct Monocular SLAM**, *J. Engel, T. Schöps, D. Cremers*, ECCV '14.
* **Semi-Dense Visual Odometry for a Monocular Camera**, *J. Engel, J. Sturm, D. Cremers*, ICCV '13.
* **DSO: Direct Sparse Odometry**, *J. Engel, V. Koltun, D. Cremers*, In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2018.

### Requirements and Dependencies

* CMake >= 3.8
* Clang >= 7.0 (or any other compilers like **GCC** or **Intel**, do not use Microsoft compilers)
* C++ 14 standard (mainly for security reasons, especially for pointers)
* C++ standard libraries, Eigen >= 3.3, OpenCV >= 3.4, [Sophus](https://github.com/strasdat/Sophus)(camera motions as Lie Group and Lie Algebra, **only** needed for estimating camera pose and scene/map reconstruction), 
Boost(**only** for multi-threading)
* [NanoGUI](https://nanogui.readthedocs.io/en/latest/) (**only** for visualization)
* [Kalibr](https://github.com/ethz-asl/kalibr) for camera calibration

### Build and Compile conventions

* **Always** build every .cpp source file as **static** libraries to lib directory
* **Always** separate .cpp and .h files into corresponding folders
* **Always** put third_party libraries or codes into third_party folder if CMake could not find automatically
* **DO NOT** git commit build (including executables) related files such as *.a, *.o, etc. Use .gitignore.
* **EXPLICITLY** enable SIMD vectorization and CPU arch optimization when compiling

### Code Style and Conventions

We follow [Google C++ Style](https://google.github.io/styleguide/cppguide.html) in general.

#### Datatype Rules/Notes
* **AVOID** use raw data types, such as arrays, raw pointers. Instead use the following:
    * **Eigen::Array** for (large) multi-dimentional data types
    * **std::vector** for address contiguous (small-medium) sequence data types
    * use **reference** for passing/returning function arguments, **always** use **smart pointers** instead of raw pointers
    * **smart pointers:**
        *  std::unique_ptr, std::make_unique
        *  std::shared_ptr, std::make_shared
    * **DO NOT** use **auto** pointers to avoid confusions
    * use raw pointers **VERY CAREFULLY** if must
* **USE** Eigen::Matrix **only** for linear algebra related storage and operations
* **USE** OpenCV cv::Mat **only** for image/camera related data storage and operations
* **USE** 32-bit float for all floating point data, use 64-bit double if must
* **USE** 32-bit int for all (contiguous) integer data
* **DO NOT** use **unsigned_int.** Use it if it is needed for interfacing with other libraries 
(**unsigned_int** has already been proved as a design flaw in C++ standard)
* **Aliasing** in Eigen: be aware if you have the same Eigen object on both side of the expression.
* **Alignment** in Eigen if you have **fixed-size vectorizable object** only.


#### Performance Concerns
* **Generally** use 32-bit aligned contiguous memory layout for all image matrices and (medium-large) data arrays (
use 128-bit or 256-bit aligned memory if certain parallel operations can be performed with SSE/AVX registers)
* **USE** fix-sized Eigen::Matrix or Eigen::Vector **only** for small matrices (total number of elements up to 16)
    * **Optional** make sure the number of elements is dividable by 4, if not padding with additional zeros
* **USE** dynamic-sized Eigen::Matrix or Eigen::Vector for **all** medium-large matrices (Eigen do memory alignment automatically for large dynamic matrices)
* **OpenCV** automatically allocate 32-bit aligned contiguous address for cv::Mat, but still check it to be sure


### TODOs

* **Test Stereo Tracking** against KITTI dataset
* **Keyframe** selection
* **Integrate Visualisation**
* **Calibrate Camera** more times, issue: OpenCV stereoRectify() output newCameraMatrix units? 
* **Asynchronous Queue** for camera/tracking interface
* **Camera Tracking** speed up: 30ms -> 10ms


# License
The source code is licensed under the GNU General Public License Version 3 (GPLv3), see http://www.gnu.org/licenses/gpl.html.




