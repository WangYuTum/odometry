// This is an I/O wrapper for camera input, running as an independent thread (constantly reading camera)
// Created by Yu Wang on 2019-01-29.

#include <io_camera.h>
#include <boost/thread.hpp>

namespace odometry
{

void RunCamera(cv::Mat* current_left, cv::Mat* current_right, std::shared_ptr<odometry::CameraPyramid> cam_ptr_left,
               std::shared_ptr<odometry::CameraPyramid> cam_ptr_right, int cam_deviceID){
  std::cout << "Camera thread id: " << boost::this_thread::get_id() << std::endl;

  // checks
  if (cam_ptr_left == nullptr || cam_ptr_right == nullptr){
    std::cout << "Invalid camera pointer in RunCamera()!" << std::endl;
    // TODO: terminate thread
  }
  cv::Mat raw_in;
  cv::VideoCapture cam_cap;
  cam_cap.open(cam_deviceID);
  if (!cam_cap.isOpened()) {
    std::cout << "ERROR! Unable to open camera\n";
    // TODO: terminate thread
  } else {
    cam_cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280); // 1280 x 1.5 = 1920 , [720, 960]
    cam_cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480); // 480 x 1.5 = 720
  }
  cv::Mat left_rect(480, 640, PixelType);
  cv::Mat right_rect(480, 640, PixelType);

  unsigned int count=0;

  while(true){
    std::cout << "read frame " << count << std::endl;
    cam_cap.read(raw_in);
    cv::Mat left_img(raw_in, cv::Rect(0, 0, 640, 480));
    cv::Mat right_img(raw_in, cv::Rect(640, 0, 640, 480));
    cam_ptr_left->UndistortRectify(left_img, left_rect);
    cam_ptr_right->UndistortRectify(right_img, right_rect);
    if (raw_in.empty()){
      std::cout << "Read frame failed!" << std::endl;
      // TODO: terminate thread
      break;
    }
    // TODO: wait for signal & assign the new captured frame
    *current_left = left_rect.clone();
    *current_right = right_rect.clone();
    boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
    count++;
  }
}


}