// The files contains all definitions regarding to depth estimation
// Created by Yu Wang on 2019-01-11.

#include <depth_estimate.h>

namespace odometry
{

// TODO: parameter list
DepthEstimator::DepthEstimator(float grad_th, float ssd_th){
  grad_th_ = grad_th;
  ssd_th_ = ssd_th;
}

GlobalStatus DepthEstimator::ComputeDepth(const cv::Mat& left_img, const cv::Mat& right_img, cv::Mat& left_val,
        cv::Mat& left_disp, cv::Mat& left_dep){
  // Note that left_val, (rectified) left_disp, left_dep are already declared and aligned to 32bit address
  if ((left_img.rows != right_img.rows) || (left_img.cols != right_img.cols)){
    std::cout << "Number of rows/cols do not match for left/right images." << std::endl;
    return -1;
  }
  if ((left_img.type() != PixelType) || right_img.type() != PixelType){
    std::cout << "Pixel type of left/right images not 32-bit float." << std::endl;
    return -1;
  }
  if ((left_img.rows != 375) || left_img.cols != 450){ // 480, 640
    std::cout << "rows != 480 or cols != 640." << std::endl;
    return -1;
  }

  // TODO: rectify left/right images
  cv::Mat left_rect = left_img;
  cv::Mat right_rect = right_img;
  // cv::Mat left_rect(480, 640, PixelType);
  // cv::Mat right_rect(480, 640, PixelType);
  // RectifyStereo(left_img, right_img, left_rect, right_rect);
  GlobalStatus disp_stat = -1;

  /************************** Strategy1 **************************/
  // Frist: compute full gradient of rectified left image
  // Second: do disparity using computed grad and depth estimation

  /************************** Strategy2 **************************/
  // One step: while looping pixels, compute gradient and do disparity search
  disp_stat = DisparityDepthEstimateStrategy2(left_rect, right_rect, left_disp, left_dep, left_val);
  if (disp_stat == -1){
    std::cout << "Depth estimation failed!" << std::endl;
    return -1;
  }
  return 0;

}

void DepthEstimator::RectifyStereo(const cv::Mat& left_img, const cv::Mat& right_img, cv::Mat& left_rect, cv::Mat& right_rect){
}

GlobalStatus DepthEstimator::DisparityDepthEstimateStrategy1(const cv::Mat& left_rect, const cv::Mat& right_rect, const cv::Mat& left_grad,
        cv::Mat& left_disp, cv::Mat& left_dep, cv::Mat& left_val){
  return -1;
}

GlobalStatus DepthEstimator::DisparityDepthEstimateStrategy2(const cv::Mat& left_rect, const cv::Mat& right_rect,
                                                     cv::Mat& left_disp, cv::Mat& left_dep, cv::Mat& left_val){

  /******** idea: set boundary start, end; loop over all pixels on the left rectified image ********/
  // a) compute grad on the pixel, if smaller than some TH continue, else
  // b) get the right boundary of epl
  // c) compute SSD value along the epl on the right rectified image, keep the smallest in a buffer
  // d) if the smallest SSD error is larger than some TH, return failed match; otherwise, set valid bool, store the value

  // check the memory
  if (!left_rect.isContinuous() || !right_rect.isContinuous() || !left_disp.isContinuous()
  || !left_dep.isContinuous() || !left_val.isContinuous()){
    std::cout << "The cv::Mat matrix is not continuous in disparity search!" << std::endl;
    return -1;
  }
  if ( (unsigned long)left_rect.ptr<float>() % 4 != 0 ||
       (unsigned long)right_rect.ptr<float>() % 4 != 0){
    std::cout << "The cv::Mat matrix is not aligned to 32-bit address in disparity search!" << std::endl;
    return -1;
  }

  float current_ssd = 0;
  float smallest_ssd = 1e+10; // initial smallest ssd err
  int match_coord = -1; // the current best match column coord
  int begin_x = 4; // skip the first 4 cols of the image
  int end_x = left_rect.cols - 4; // skp the last 4 cols of the image, 640-4
  int begin_y = 4; // skip the first 4 rows of the image
  int end_y = left_rect.rows - 4; // skp the last 4 rows of the image, 480-4
  const float* left_pp_row_ptr = nullptr;
  const float* left_p_row_ptr = nullptr;
  const float* left_row_ptr = nullptr;
  const float* left_n_row_ptr = nullptr;
  const float* left_nn_row_ptr = nullptr;
  const float* right_pp_row_ptr = nullptr;
  const float* right_p_row_ptr = nullptr;
  const float* right_row_ptr = nullptr;
  const float* right_n_row_ptr = nullptr;
  const float* right_nn_row_ptr = nullptr;
  uint8_t* left_val_row_ptr = nullptr;
  float* left_disp_row_ptr = nullptr;
  float grad_x = 0;
  float grad_y = 0;
  float mag_grad = 0;
  __m256 left_pattern;

  std::cout << "start computing ..." << std::endl;
  clock_t begin = clock();
  for (int y=begin_y; y<end_y; y++){
    // get pointers
    left_p_row_ptr = left_rect.ptr<float>(y-1);
    left_row_ptr = left_rect.ptr<float>(y);
    left_n_row_ptr = left_rect.ptr<float>(y+1);
    for (int x=begin_x; x<end_x; x++){
      // compute magnitude grad of current pixel using central difference
      grad_x = 0.5f * (*(left_row_ptr+x+1)- *(left_row_ptr+x-1));
      grad_y = 0.5f * (*(left_p_row_ptr+x) - *(left_n_row_ptr+x));
      mag_grad = std::sqrt(grad_x*grad_x + grad_y*grad_y);
      if (mag_grad<grad_th_) continue;
      else {
        // get pointers for left_val, left_disp
        left_val_row_ptr = left_val.ptr<uint8_t>(y);
        left_disp_row_ptr = left_disp.ptr<float>(y);
        left_pp_row_ptr = left_rect.ptr<float>(y-2);
        left_nn_row_ptr = left_rect.ptr<float>(y+2);
        // now we do the actuall disparity match on the right image epl, get the pointers
        right_row_ptr = right_rect.ptr<float>(y);
        right_p_row_ptr = right_rect.ptr<float>(y-1);
        right_pp_row_ptr = right_rect.ptr<float>(y-2);
        right_n_row_ptr = right_rect.ptr<float>(y+1);
        right_nn_row_ptr = right_rect.ptr<float>(y+2);
        smallest_ssd = 1e+10;
        // loop the search range
        /***************** Search along epl: Naive implementation **************/
        /*
        for (int right_x=begin_x; right_x<x; right_x++){
          // compute ssd
          current_ssd = ComputeSsdDso(left_pp_row_ptr, left_p_row_ptr, left_row_ptr, left_n_row_ptr, left_nn_row_ptr,
                                      right_pp_row_ptr, right_p_row_ptr, right_row_ptr, right_n_row_ptr, right_nn_row_ptr, x, right_x);
          match_coord = (current_ssd < smallest_ssd) ? (right_x) : match_coord;
          smallest_ssd = (current_ssd < smallest_ssd) ? (current_ssd) : smallest_ssd;
        } // loop right cols
        */
        /***************** Search along epl: SSE implementation **************/
        left_pattern = _mm256_set_ps(*(left_pp_row_ptr+x), *(left_p_row_ptr+x-1), *(left_p_row_ptr+x+1), *(left_row_ptr+x-2),
                                     *(left_row_ptr+x), *(left_row_ptr+x+2), *(left_n_row_ptr+x-1), *(left_nn_row_ptr+x));
        for (int right_x=begin_x; right_x<x; right_x++){
          ComputeSsdDsoSse(left_pattern, right_pp_row_ptr, right_p_row_ptr, right_row_ptr, right_n_row_ptr,
                  right_nn_row_ptr, right_x, &current_ssd);
          match_coord = (current_ssd < smallest_ssd) ? (right_x) : match_coord;
          smallest_ssd = (current_ssd < smallest_ssd) ? (current_ssd) : smallest_ssd;
        } // loop right cols
        if (smallest_ssd > ssd_th_)
          continue;
        else {
          *(left_val_row_ptr+x) = 1; //left_val.at<uint8_t>(y, x) = 1;
          *(left_disp_row_ptr+x) = std::abs(x-match_coord); // left_disp.at<float>(y, x) = std::abs(x-match_coord);
        } // a successful match, store the disparity value, set valid mask
      } // if left grad is large
    } // loop left cols
  } // loop left rows
  clock_t end = clock();
  std::cout << "eval disparity: " << double(end - begin) / CLOCKS_PER_SEC * 1000.0f << " ms." << std::endl;
  return 0;
}

inline float DepthEstimator::ComputeSsd5x5(const float* left_pp_row_ptr, const float* left_p_row_ptr, const float* left_row_ptr, const float* left_n_row_ptr, const float* left_nn_row_ptr,
                        const float* right_pp_row_ptr, const float* right_p_row_ptr, const float* right_row_ptr, const float* right_n_row_ptr, const float* right_nn_row_ptr,
                        int left_x, int right_x){
  float sum = 0;
  for (int delta_x = -2; delta_x < 3; delta_x++)
    sum += std::pow(*(left_pp_row_ptr+left_x+delta_x) - *(right_pp_row_ptr+right_x+delta_x), 2);
  for (int delta_x = -2; delta_x < 3; delta_x++)
    sum += std::pow(*(left_p_row_ptr+left_x+delta_x) - *(right_p_row_ptr+right_x+delta_x), 2);
  for (int delta_x = -2; delta_x < 3; delta_x++)
    sum += std::pow(*(left_row_ptr+left_x+delta_x) - *(right_row_ptr+right_x+delta_x), 2);
  for (int delta_x = -2; delta_x < 3; delta_x++)
    sum += std::pow(*(left_n_row_ptr+left_x+delta_x) - *(right_n_row_ptr+right_x+delta_x), 2);
  for (int delta_x = -2; delta_x < 3; delta_x++)
    sum += std::pow(*(left_nn_row_ptr+left_x+delta_x) - *(right_nn_row_ptr+right_x+delta_x), 2);
  return sum;
}

inline float DepthEstimator::ComputeSsdDso(const float* left_pp_row_ptr, const float* left_p_row_ptr, const float* left_row_ptr, const float* left_n_row_ptr, const float* left_nn_row_ptr,
                           const float* right_pp_row_ptr, const float* right_p_row_ptr, const float* right_row_ptr, const float* right_n_row_ptr, const float* right_nn_row_ptr,
                           int left_x, int right_x){
  float sum = 0;
  sum += std::pow(*(left_pp_row_ptr+left_x) - *(right_pp_row_ptr+right_x), 2);
  sum += std::pow(*(left_p_row_ptr+left_x-1) - *(right_p_row_ptr+right_x-1), 2);
  sum += std::pow(*(left_p_row_ptr+left_x+1) - *(right_p_row_ptr+right_x+1), 2);
  sum += std::pow(*(left_row_ptr+left_x-2) - *(right_row_ptr+right_x-2), 2);
  sum += std::pow(*(left_row_ptr+left_x) - *(right_row_ptr+right_x), 2);
  sum += std::pow(*(left_row_ptr+left_x+2) - *(right_row_ptr+right_x+2), 2);
  sum += std::pow(*(left_n_row_ptr+left_x-1) - *(right_n_row_ptr+right_x-1), 2);
  sum += std::pow(*(left_nn_row_ptr+left_x) - *(right_nn_row_ptr+right_x), 2);
  return sum;
}

inline void DepthEstimator::ComputeSsdDsoSse(const __m256& left_pattern, const float* right_pp_row_ptr, const float* right_p_row_ptr,
        const float* right_row_ptr, const float* right_n_row_ptr, const float* right_nn_row_ptr, int x, float* result){
  __m256 right_pattern;
  __m256 sub_pattern;
  __m256 sq_pattern;
  __m256 sum_sq;
  __m128 low_sum, high_sum, sum_low_high, pack_sum, inter_sum;
  right_pattern = _mm256_set_ps(*(right_pp_row_ptr+x), *(right_p_row_ptr+x-1), *(right_p_row_ptr+x+1), *(right_row_ptr+x-2),
                               *(right_row_ptr+x), *(right_row_ptr+x+2), *(right_n_row_ptr+x-1), *(right_nn_row_ptr+x));
  sub_pattern = _mm256_sub_ps(left_pattern, right_pattern);
  sq_pattern = _mm256_mul_ps(sub_pattern, sub_pattern);
  sum_sq = _mm256_hadd_ps(sq_pattern, sq_pattern); // [0,1,4,5]
  low_sum = _mm256_extractf128_ps(sum_sq, 0);
  high_sum = _mm256_extractf128_ps(sum_sq, 1);
  sum_low_high = _mm_hadd_ps(low_sum, high_sum);
  inter_sum = _mm_movehl_ps(sum_low_high, sum_low_high);
  pack_sum = _mm_add_ps(sum_low_high, inter_sum);
  _mm_store_ss(result, pack_sum);
}

inline float DepthEstimator::ComputeSsdLine(const float* left_row_ptr, const float* right_row_ptr, int left_x, int right_x){
  float sum = 0;
  sum += std::pow(*(left_row_ptr+left_x-2) - *(right_row_ptr+right_x-2), 2);
  sum += std::pow(*(left_row_ptr+left_x-1) - *(right_row_ptr+right_x-1), 2);
  sum += std::pow(*(left_row_ptr+left_x) - *(right_row_ptr+right_x), 2);
  sum += std::pow(*(left_row_ptr+left_x+1) - *(right_row_ptr+right_x+1), 2);
  sum += std::pow(*(left_row_ptr+left_x+2) - *(right_row_ptr+right_x+2), 2);
  return sum;
}

} // namespace odometry

