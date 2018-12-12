cmake_minimum_required(VERSION 3.8)
project(odometry)

# -> required packages
find_package(Eigen3 3.3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV 3.4 REQUIRED)
include_directories(${OPENCV_INCLUDE_DIR})
# <- required packages

# -> set language and compiler optimization specific flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W -Wcast-qual -Wwrite-strings -Wextra") # -Werror
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mtune=haswell -march=haswell -m64 -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx2")
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
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/Sophus/sophus)
# <- include headers

# -> MACROS related to dataset
add_definitions(-DHOME_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/dataset/")
# <- MACROS

# -> build libs
add_library(image_processing_global STATIC src/image_processing_global.cpp)
add_library(image_pyramid STATIC src/image_pyramid.cpp)
add_library(lm_optimizer STATIC src/lm_optimizer.cpp)
# <- build libs

# -> build executable
add_executable(test_optimizer test_optimizer.cpp)
# <- build executable

# -> link
target_link_libraries(test_optimizer image_processing_global image_pyramid lm_optimizer opencv_core opencv_imgcodecs opencv_highgui)
# <- link






