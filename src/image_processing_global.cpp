// Created by Yu Wang on 04.12.18.
// This is an implementation of functions defined in ODOMETRY_IMAGE_PROCESSING_GLOBAL_H

#include <image_processing_global.h>
#include <iostream>
#include <immintrin.h> // AVX instruction set


namespace odometry
{

GlobalStatus GaussianImagePyramidNaive(int num_levels, const cv::Mat& in_img, std::vector<cv::Mat>& out_pyramids, bool smooth){
  int rows = in_img.rows;
  int cols = in_img.cols;
  int channels = in_img.channels();
  // necessary checks
  if (rows % 2 != 0 || cols % 2 !=0 || channels != 1 || in_img.type() != PixelType){
    // TODO: uncomment the following
//    std::cout << "Original image rows/cols are not even OR channels != 2 OR pixeltype is not CV_32F(float)! Create image pyramids failed." << std::endl;
//    std::cout << "Number of rows: " << rows << std::endl;
//    std::cout << "Number of cols: " << cols << std::endl;
//    std::cout << "Number of channels: " << channels << std::endl;
//    std::cout << "Original image type: " << in_img.type() << std::endl;
//      return -1;
  }

  // smooth the original image using gaussian kernel as the level-0 pyramid
  out_pyramids.emplace_back(cv::Mat(rows, cols, PixelType));
  if (smooth){
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

GlobalStatus MedianDepthPyramidNaive(int num_levels, const cv::Mat& in_img, std::vector<cv::Mat>& out_pyramids, bool smooth){
  int rows = in_img.rows;
  int cols = in_img.cols;
  int channels = in_img.channels();
  cv::Scalar init_val(0);
  // necessary checks
  if (rows % 2 != 0 || cols % 2 !=0 || channels != 1 || in_img.type() != PixelType){
    // TODO: uncomment the following
//    std::cout << "Original depth rows/cols are not even OR channels != 2 OR pixeltype is not CV_32F(float)! Create depth pyramids failed." << std::endl;
//    std::cout << "Number of rows: " << rows << std::endl;
//    std::cout << "Number of cols: " << cols << std::endl;
//    std::cout << "Number of channels: " << channels << std::endl;
//    std::cout << "Original depth type: " << in_img.type() << std::endl;
//    return -1;
  }

  // smooth the original image using median filter, take care of Invalid depth value(0)
  out_pyramids.emplace_back(cv::Mat(rows, cols, PixelType, init_val));
  if (smooth){
    cv::medianBlur(in_img, out_pyramids[0], 3);
  } else{
    in_img.copyTo(out_pyramids[0]);
  }
  // level-1 pyramid, since we assume level-0 is already smoothed we only do downsampling by ignoring even-numbered rows & cols
  rows = rows / 2;
  cols = cols / 2;
  out_pyramids.emplace_back(cv::Mat(rows, cols, PixelType, init_val));
  for (int y = 0; y < rows; y++){
    for (int x = 0; x < cols; x++){
      out_pyramids[1].at<float>(y, x) = out_pyramids[0].at<float>(y*2+1, x*2+1);
    }
  }

  // down sample images: smooth by median filter, then downsample by ignoring even-numbered rows & cols
  rows = rows / 2;
  cols = cols / 2;
  for (int l = 2; l < num_levels; l++){
    out_pyramids.emplace_back(cv::Mat(rows, cols, PixelType, init_val));
    // cv::Mat smooth_out;
    // cv::medianBlur(out_pyramids[l-1], smooth_out, 3);
    for (int y = 0; y < rows; y++){
      for (int x = 0; x < cols; x++){
        out_pyramids[l].at<float>(y, x) = out_pyramids[l-1].at<float>(y*2+1, x*2+1);
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

// sse implementation of depth pyramid: optimze for down-sampling & memory operations
GlobalStatus MedianDepthPyramidSse(int num_levels, const cv::Mat& in_img, std::vector<cv::Mat>& out_pyramids, bool smooth){
  // Note that input image array and output image array MUST all be contiguous and algined against 32 bits (single precision)
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
    // TODO: return -1
    //return -1;
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
      PyramidDownSse(out_pyramids[0], out_pyramids[1], rows, cols);
    }
  }

  // down sample images: smooth by median filter, then downsample by ignoring even-numbered rows & cols
  rows = rows / 2;
  cols = cols / 2;
  for (int l = 2; l < num_levels; l++){
    out_pyramids.emplace_back(cv::Mat(rows, cols, PixelType));
    cv::Mat smooth_out;
    cv::medianBlur(out_pyramids[l-1], smooth_out, 3);
    PyramidDownSse(smooth_out, out_pyramids[l], rows, cols);
    rows = rows / 2;
    cols = cols / 2;
  }
  if (out_pyramids.size() != num_levels){
    std::cout << "Depth Pyramid size != num_levels. " << std::endl;
    return -1;
  }

  return 0;
}

void PyramidDownSse(cv::Mat& in_img, cv::Mat& out_img, int rows, int cols){
  // check memeory layout
  if (!in_img.isContinuous() || !out_img.isContinuous()){
    std::cout << "Input/Output img not continuous!" << std::endl;
    exit(-1);
  }
  // define pointers and registers
  float* in_row_ptr;
  float* out_row_ptr;
  int num_blocks;
  __m256 tmp_reg[5]; // use 5 registers as a group, 40 floating data in total
  for (int row_idx = 0; row_idx < rows; row_idx++){
    in_row_ptr = in_img.ptr<float>(row_idx*2 + 1);
    out_row_ptr = out_img.ptr<float>(row_idx);
    num_blocks = cols / 40;
    for (int block_id = 0; block_id < num_blocks; block_id++){
      // load data from input image
      tmp_reg[0] = _mm256_set_ps (*(in_row_ptr+block_id*40*2+0*16+1),
                                  *(in_row_ptr+block_id*40*2+0*16+3),
                                  *(in_row_ptr+block_id*40*2+0*16+5),
                                  *(in_row_ptr+block_id*40*2+0*16+7),
                                  *(in_row_ptr+block_id*40*2+0*16+9),
                                  *(in_row_ptr+block_id*40*2+0*16+11),
                                  *(in_row_ptr+block_id*40*2+0*16+13),
                                  *(in_row_ptr+block_id*40*2+0*16+15));
      tmp_reg[1] = _mm256_set_ps (*(in_row_ptr+block_id*40*2+1*16+1),
                                  *(in_row_ptr+block_id*40*2+1*16+3),
                                  *(in_row_ptr+block_id*40*2+1*16+5),
                                  *(in_row_ptr+block_id*40*2+1*16+7),
                                  *(in_row_ptr+block_id*40*2+1*16+9),
                                  *(in_row_ptr+block_id*40*2+1*16+11),
                                  *(in_row_ptr+block_id*40*2+1*16+13),
                                  *(in_row_ptr+block_id*40*2+1*16+15));
      tmp_reg[2] = _mm256_set_ps (*(in_row_ptr+block_id*40*2+2*16+1),
                                  *(in_row_ptr+block_id*40*2+2*16+3),
                                  *(in_row_ptr+block_id*40*2+2*16+5),
                                  *(in_row_ptr+block_id*40*2+2*16+7),
                                  *(in_row_ptr+block_id*40*2+2*16+9),
                                  *(in_row_ptr+block_id*40*2+2*16+11),
                                  *(in_row_ptr+block_id*40*2+2*16+13),
                                  *(in_row_ptr+block_id*40*2+2*16+15));
      tmp_reg[3] = _mm256_set_ps (*(in_row_ptr+block_id*40*2+3*16+1),
                                  *(in_row_ptr+block_id*40*2+3*16+3),
                                  *(in_row_ptr+block_id*40*2+3*16+5),
                                  *(in_row_ptr+block_id*40*2+3*16+7),
                                  *(in_row_ptr+block_id*40*2+3*16+9),
                                  *(in_row_ptr+block_id*40*2+3*16+11),
                                  *(in_row_ptr+block_id*40*2+3*16+13),
                                  *(in_row_ptr+block_id*40*2+3*16+15));
      tmp_reg[4] = _mm256_set_ps (*(in_row_ptr+block_id*40*2+4*16+1),
                                  *(in_row_ptr+block_id*40*2+4*16+3),
                                  *(in_row_ptr+block_id*40*2+4*16+5),
                                  *(in_row_ptr+block_id*40*2+4*16+7),
                                  *(in_row_ptr+block_id*40*2+4*16+9),
                                  *(in_row_ptr+block_id*40*2+4*16+11),
                                  *(in_row_ptr+block_id*40*2+4*16+13),
                                  *(in_row_ptr+block_id*40*2+4*16+15));
      // store data to output image
      _mm256_store_ps (out_row_ptr+block_id*40+0*8, tmp_reg[0]);
      _mm256_store_ps (out_row_ptr+block_id*40+1*8, tmp_reg[1]);
      _mm256_store_ps (out_row_ptr+block_id*40+2*8, tmp_reg[2]);
      _mm256_store_ps (out_row_ptr+block_id*40+3*8, tmp_reg[3]);
      _mm256_store_ps (out_row_ptr+block_id*40+4*8, tmp_reg[4]);
    }
  }
}


// TODO, using native c++ for loop combined with openmp to warp entire image
void WarpImageNative(const cv::Mat& img_in, const Affine4f& kTransMat, cv::Mat& warped_img){

}

// TODO, using sse to warp entire image
void WarpImageSse(const cv::Mat& img_in, const Affine4f& kTransMat, cv::Mat& warped_img){

}



} // namespace odometry

