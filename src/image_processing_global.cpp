// Created by Yu Wang on 04.12.18.
// This is an implementation of functions defined in ODOMETRY_IMAGE_PROCESSING_GLOBAL_H

#include <image_processing_global.h>
#include <iostream>


namespace odometry
{

GlobalStatus GaussianImagePyramidNaive(int num_levels, const cv::Mat& in_img, std::vector<cv::Mat>& out_pyramids, bool smooth){
  int rows = in_img.rows;
  int cols = in_img.cols;
  int channels = in_img.channels();
  // necessary checks
  if (rows % 2 != 0 || cols % 2 !=0 || channels != 1 || in_img.type() != PixelType){
    std::cout << "Original image rows/cols are not even OR channels != 2 OR pixeltype is not CV_32F(float)! Create image pyramids failed." << std::endl;
    std::cout << "Number of rows: " << rows << std::endl;
    std::cout << "Number of cols: " << cols << std::endl;
    std::cout << "Number of channels: " << channels << std::endl;
    std::cout << "Original image type: " << in_img.type() << std::endl;
    return -1;
  }

  // smooth the original image using gaussian kernel as the level-0 pyramid
  out_pyramids.emplace_back(cv::Mat(rows, cols, PixelType));
  if (smooth == true){
    cv::GaussianBlur(in_img, out_pyramids[0], cv::Size(3, 3), 0);
  } else{
    in_img.copyTo(out_pyramids[0]);
  }
  // level-1 pyramid, since we assume level-0 is already smoothed and we want to avoid smoothing again
  rows = rows / 2;
  cols = cols / 2;
  out_pyramids.emplace_back(cv::Mat(rows, cols, PixelType));
  cv::pyrDown(in_img, out_pyramids[1], cv::Size(cols, rows));

  // downsample images by first convolving with gaussian kernel, then remove even-numbered rows&cols
  rows = rows / 2;
  cols = cols / 2;

  for (int l = 2; l < num_levels; l++){
    out_pyramids.emplace_back(cv::Mat(rows, cols, PixelType));
    cv::pyrDown(out_pyramids[l-1], out_pyramids[l], cv::Size(cols, rows));
    rows = rows / 2;
    cols = cols / 2;
  }
  if (out_pyramids.size() != num_levels){
    std::cout << "Image Pyramid size != num_levels. " << std::endl;
    return -1;
  }

  return 0;
}

GlobalStatus GaussianDepthPyramidNaive(int num_levels, const cv::Mat& in_img, std::vector<cv::Mat>& out_pyramids, bool smooth){
  int rows = in_img.rows;
  int cols = in_img.cols;
  int channels = in_img.channels();
  // necessary checks
  if (rows % 2 != 0 || cols % 2 !=0 || channels != 1 || in_img.type() != PixelType){
    std::cout << "Original depth rows/cols are not even OR channels != 2 OR pixeltype is not CV_32F(float)! Create depth pyramids failed." << std::endl;
    std::cout << "Number of rows: " << rows << std::endl;
    std::cout << "Number of cols: " << cols << std::endl;
    std::cout << "Number of channels: " << channels << std::endl;
    std::cout << "Original depth type: " << in_img.type() << std::endl;
    return -1;
  }

  // smooth the original image using median filter, take care of Invalid depth value(0)
  out_pyramids.emplace_back(cv::Mat(rows, cols, PixelType));
  if (smooth == true){
    cv::medianBlur(in_img, out_pyramids[0], 3);
  } else{
    in_img.copyTo(out_pyramids[0]);
  }
  // level-1 pyramid, since we assume level-0 is already smoothed we only do downsampling by ignoring even-numbered rows & cols
  rows = rows / 2;
  cols = cols / 2;
  out_pyramids.emplace_back(cv::Mat(rows, cols, PixelType));
  for (int y = 0; y < rows; y++){
    for (int x = 0; x < cols; x++){
      out_pyramids[1].at<float>(y, x) = out_pyramids[0].at<float>(y*2, x*2); // y*2+1, x*2+1
    }
  }

  // down sample images: smooth by median filter, then downsample by ignoring even-numbered rows & cols
  rows = rows / 2;
  cols = cols / 2;
  for (int l = 2; l < num_levels; l++){
    out_pyramids.emplace_back(cv::Mat(rows, cols, PixelType));
    cv::Mat smooth_out;
    cv::medianBlur(out_pyramids[l-1], smooth_out, 3);
    for (int y = 0; y < rows; y++){
      for (int x = 0; x < cols; x++){
        out_pyramids[l].at<float>(y, x) = smooth_out.at<float>(y*2+1, x*2+1);
      }
    }
    rows = rows / 2;
    cols = cols / 2;
  }
  if (out_pyramids.size() != num_levels){
    std::cout << "Depth Pyramid size != num_levels. " << std::endl;
    return -1;
  }

  return 0;
}


// TODO, using native c++ for loop combined with openmp to warp entire image
void WarpImageNative(const cv::Mat& img_in, const Matrix44f& kTransMat, cv::Mat& warped_img){

}

// TODO, using sse to warp entire image
void WarpImageSse(const cv::Mat& img_in, const Matrix44f& kTransMat, cv::Mat& warped_img){

}

// TODO: sse
GlobalStatus GaussianImagePyramidSse(int num_levels, const cv::Mat& in_img, std::vector<cv::Mat>& out_pyramids, bool smooth){

}

// TODO: sse
GlobalStatus GaussianDepthPyramidSse(int num_levels, const cv::Mat& in_img, std::vector<cv::Mat>& out_pyramids, bool smooth){

}



} // namespace odometry

