cmake_minimum_required(VERSION 3.8)
project(odometry)

add_compile_definitions(EIGEN_STACK_ALLOCATION_LIMIT=0)

# -> required packages
find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV 3.4 REQUIRED)
find_package(Boost 1.68 COMPONENTS thread date_time system chrono REQUIRED)

#add_subdirectory(third_party/nanogui)
#set(NANOGUI_BUILD_EXAMPLE OFF)
#set(NANOGUI_BUILD_PYTHON OFF)
#set(NANOGUI_INSTALL OFF)
# <- required packages

# -> definitions
#add_definitions(${NANOGUI_EXTRA_DEFS})
# <- definitions

# -> set language and compiler optimization specific flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W -Wcast-qual -Wwrite-strings -Wextra") # -Werror
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=haswell -march=haswell -m64 -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx2 -mavx")
# <- set language and compiler optimization specific flags

# -> build config
if (NOT(EXISTS ${PROJECT_SOURCE_DIR}/bin AND IS_DIRECTORY ${PROJECT_SOURCE_DIR}/bin))
    file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
endif (NOT(EXISTS ${PROJECT_SOURCE_DIR}/bin AND IS_DIRECTORY ${PROJECT_SOURCE_DIR}/bin))
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

if (NOT(EXISTS ${PROJECT_SOURCE_DIR}/lib AND IS_DIRECTORY ${PROJECT_SOURCE_DIR}/lib))
    file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
endif (NOT(EXISTS ${PROJECT_SOURCE_DIR}/lib AND IS_DIRECTORY ${PROJECT_SOURCE_DIR}/lib))
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(BUILD_SHARED_LIBS ON)
# <- build config

# -> include headers
include_directories(${OPENCV_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/Sophus/sophus)
#include_directories(${PROJECT_SOURCE_DIR}/third_party/nanogui/include)
#include_directories(${NANOGUI_EXTRA_INCS})
# <- include headers

# -> MACROS related to dataset for tracking only
add_definitions(-DHOME_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/dataset/")
# <- MACROS

# -> build libs
add_library(image_processing_global STATIC src/image_processing_global.cpp)
add_library(image_pyramid STATIC src/image_pyramid.cpp)
add_library(lm_optimizer STATIC src/lm_optimizer.cpp)
add_library(depth_estimate STATIC src/depth_estimate.cpp)
add_library(camera STATIC src/camera.cpp)
add_library(io_camera STATIC src/io_camera.cpp)
#add_library(vis STATIC src/vis.cpp)
#add_library(vis_screen STATIC src/vis_screen.cpp)
# <- build libs

# -> build executable
#add_executable(test_optimizer test_optimizer.cpp)
#add_executable(test_disparity test_disparity.cpp)
#add_executable(test_camera_setup test_camera_setup.cpp)
#add_executable(run_odometry_kitti run_odometry_kitti_offline.cpp)
add_executable(run_odometry_live run_odometry_live.cpp)
# <- build executable

# -> link
#target_link_libraries(test_optimizer image_processing_global image_pyramid lm_optimizer opencv_core opencv_imgcodecs opencv_highgui)
#target_link_libraries(test_disparity depth_estimate opencv_core opencv_imgcodecs opencv_highgui opencv_photo camera)
#target_link_libraries(test_camera_setup opencv_core camera opencv_imgproc opencv_calib3d)
#target_link_libraries(run_odometry_kitti camera depth_estimate image_processing_global image_pyramid lm_optimizer opencv_core opencv_imgcodecs opencv_imgproc opencv_highgui opencv_photo opencv_calib3d)
target_link_libraries(run_odometry_live camera depth_estimate image_processing_global image_pyramid lm_optimizer
opencv_core opencv_imgcodecs opencv_imgproc opencv_highgui opencv_photo opencv_calib3d opencv_videoio io_camera
        boost_date_time boost_system boost_chrono boost_thread-mt)
# <- link






