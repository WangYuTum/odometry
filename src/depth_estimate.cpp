// The files contains all definitions regarding to depth estimation
// Created by Yu Wang on 2019-01-11.

#include <depth_estimate.h>

namespace odometry
{

DepthEstimator::DepthEstimator(float grad_th, float ssd_th, float photo_th, float min_depth, float max_depth,
        float lambda, float huber_delta, float precision, int max_iters, int boundary, const std::shared_ptr<CameraPyramid>& left_cam_ptr,
                               const std::shared_ptr<CameraPyramid>& right_cam_ptr, float baseline,  int max_residuals=5000){
  grad_th_ = grad_th;
  ssd_th_ = ssd_th;
  photo_th_ = photo_th;
  min_depth_ = min_depth;
  max_depth_ = max_depth;
  lambda_ = lambda;
  precision_ = precision;
  max_iters_ = max_iters;
  boundary_ = boundary;
  camera_ptr_left_ = left_cam_ptr;
  camera_ptr_right_ = right_cam_ptr;
  baseline_ = baseline;
  max_residuals_ = max_residuals;
  huber_delta_ = huber_delta;
}

DepthEstimator::~DepthEstimator(){
  camera_ptr_left_.reset();
  camera_ptr_right_.reset();
}

GlobalStatus DepthEstimator::ComputeDepth(const cv::Mat& left_img, const cv::Mat& right_img, cv::Mat& left_val,
        cv::Mat& left_disp, cv::Mat& left_dep){

  // Note that left_val, (rectified) left_disp, left_dep are already initialised and aligned to 32bit address
  if ((left_img.rows != right_img.rows) || (left_img.cols != right_img.cols)){
    std::cout << "Number of rows/cols do not match for left/right images." << std::endl;
    return -1;
  }
  if ((left_img.type() != CV_32F) || right_img.type() != CV_32F){
    std::cout << "Pixel type of left/right images not 32-bit float." << std::endl;
    return -1;
  }
  // TODO: check image size
  if ((left_img.rows != 376) || left_img.cols != 1241){ // 480, 640
    std::cout << "rows != 480 or cols != 640." << std::endl;
    return -1;
  }

  // loop for each pixel, compute gradient and do disparity search
  GlobalStatus disp_stat = -1;
  GlobalStatus opt_stat = -1;
  clock_t begin, end;
  std::cout << "computing disparity ..." << std::endl;
  begin = clock();
  disp_stat = DisparityDepthEstimate(left_img, right_img, left_disp, left_dep, left_val);
  if (disp_stat == -1){
    std::cout << "Disparity search failed!" << std::endl;
    return -1;
  } else {
    std::cout << "valid disparities: " << cv::sum(left_val)[0] << std::endl;
  }

  // depth optimization after initial disparity search
  std::cout << "optimizing depth ..." << std::endl;
  opt_stat = DepthOptimization(left_img, right_img, left_dep, left_val);
  end = clock();
  std::cout << "end optimization: " << double(end - begin) / CLOCKS_PER_SEC * 1000.0f << " ms." << std::endl;
  if (opt_stat == -1){
    std::cout << "Depth optimization failed!" << std::endl;
    return -1;
  } else {
    std::cout << "valid depth: " << cv::sum(left_val)[0] << std::endl;
  }

  return 0;
}

GlobalStatus DepthEstimator::DepthOptimization(const cv::Mat& left_rect, const cv::Mat& right_rect,
        cv::Mat& left_dep, cv::Mat& left_val){

  /******** idea: accumulate residuals/jacobian on valid points and minimize re-projection error ********/
  // NOTEs:
  //  * use LM (with Huber lose)
  //  * remove points that have large photometric error after optimization terminates
  //  * remove points that have too small/large depth values after optimization terminates
  //  * keep input points under 4000~, output 80% points as final valid result

  // find all valid pixels with inverse depth values and put them into an Eigen::Matrix,
  int num_residuals = 0;
  float current_lambda = lambda_;
  int iter_count = 0;
  float err_last = 1e+10;
  float err_now = 0.0;
  float err_diff = 1e+10;
  GlobalStatus compute_status;
  Eigen::Matrix<float, Eigen::Dynamic, 1> init_depth; // copied from left_dep, but only valid ones [z0, z1, ...]
  Eigen::Matrix<float, Eigen::Dynamic, 1> current_depth; // the current best estimate of inverse depth
  Eigen::Matrix<float, Eigen::Dynamic, 1> pre_depth; // previous best estimate of inverse depth
  Eigen::Matrix<float, Eigen::Dynamic, 1> tmp_depth; // attempted update
  Eigen::Matrix<float, Eigen::Dynamic, 1> delta_depth; // delta
  Eigen::Matrix<float, Eigen::Dynamic, 1> residuals; // keep them to filter pixels with large photometric error
  std::vector<std::pair<int, int>> coord_valid; // the coordinates of valid pixels on left image [(x0,y0), (x1,y1), ...]
  init_depth.resize(max_residuals_, 1); // resize to [max_residuals_,1], max_residuals_=10000 by default
  for (int y = 0; y < left_rect.rows; y++){
    for (int x = 0; x < left_rect.cols; x++){
      if (left_val.at<uint8_t>(y, x) == 1){
        init_depth.row(num_residuals) << left_dep.at<float>(y, x);
        coord_valid.emplace_back(std::make_pair(x, y));
        num_residuals++;
      }
    }
  }
  //std::cout << "number of residuals: " << num_residuals << std::endl;
  init_depth.conservativeResize(num_residuals, 1);
  Eigen::DiagonalMatrix<float, Eigen::Dynamic> jtwj; // jtwj, diagonal
  Eigen::DiagonalMatrix<float, Eigen::Dynamic> A; // jtwj + lambda*jtwj, diagonal
  Eigen::Matrix<float, Eigen::Dynamic, 1> b; // -jtwr
  current_depth.setZero(num_residuals, 1);
  current_depth = init_depth; // set current best to initial values
  residuals.setZero(num_residuals, 1); // [N,1]
  pre_depth.setZero(num_residuals, 1); // [N,1]
  delta_depth.setZero(num_residuals, 1); // [N,1]
  tmp_depth.setZero(num_residuals, 1); // [N,1]
  jtwj.setIdentity(num_residuals); // set jtwj to identity diagonal of size [N,N]
  A.setIdentity(num_residuals); // set A to identity diagonal of size [N,N]
  b.setZero(num_residuals, 1); // set b to zeros of size [N,1]

  // now we have:
  //  * initial depth, init_depth: [N,1]
  //  * current best, current_depth: [N,1]
  //  * previous best, pre_depth: [N,1]
  //  * delta, delta_depth: [N,1]
  //  * A: [N,N] diagonal identity
  //  * b: [N,1] zeros
  tmp_depth = current_depth;
  iters_stat_ = 0;
  cost_stat_ = 0;
  //std::cout << "start iterating ..." << std::endl;
  while(max_iters_ > iter_count){
    // compute new_err, jtwj, b using tmp_depth
    //std::cout << "iter# " << iter_count << std::endl;
    compute_status = ComputeResidualJacobian(tmp_depth, coord_valid, jtwj, b, err_now, num_residuals, left_rect, right_rect, residuals);
    if (compute_status == -1){
      std::cout << "Evaluate Residual & Jacobian failed " << std::endl;
      return -1;
    }
    // if new_err > last_err: increase lambda, set current_depth to pre_depth
    if (err_now > err_last){ // bad depth estimate, do not update
      current_lambda = current_lambda * 10.0f;
      if (current_lambda > 1e+5) { break; }
      current_depth = pre_depth;
    } else{ // good depth estimate -> update, save previous
      current_depth = tmp_depth;
      pre_depth = current_depth;
      err_diff = err_now / err_last;
      if (err_diff > precision_) { break; }
      err_last = err_now;
      current_lambda = std::max(current_lambda / 10.0f, float(1e-7));
    }
    //std::cout << "solving system ..." << std::endl;
    // solve system to get delta, set tmp_depth = delta + current_depth
    A.diagonal() = jtwj.diagonal() + current_lambda * jtwj.diagonal();
    delta_depth = A.inverse() * b;  // numerically stable since A is diagonal
    tmp_depth = delta_depth + current_depth;
    iter_count++;
  }
  //std::cout << "end iterating." << std::endl;
  iters_stat_ = iter_count;
  cost_stat_ = err_now;

  // Assign the computed inverse depth to left_dep, and update left_val:
  //  * remove points that have large photometric error after optimization terminates
  //  * remove points that have too small/large depth values after optimization terminates
  for (int i = 0; i < num_residuals; i++){
    // the residual is either too large or is invalid
    if (residuals(i, 0) > photo_th_ || residuals(i, 0) == -1000){
      left_val.at<uint8_t>(coord_valid[i].second, coord_valid[i].first) = 0;
      left_dep.at<float>(coord_valid[i].second, coord_valid[i].first) = 0;
    } else {
      // if the optimized depth is beyond some interval, set it invalid
      if (1.0f / current_depth(i, 0) > max_depth_ || 1.0f / current_depth(i, 0) < min_depth_){
        left_val.at<uint8_t>(coord_valid[i].second, coord_valid[i].first) = 0;
        left_dep.at<float>(coord_valid[i].second, coord_valid[i].first) = 0;
      } else {
        left_val.at<uint8_t>(coord_valid[i].second, coord_valid[i].first) = 1;
        left_dep.at<float>(coord_valid[i].second, coord_valid[i].first) = current_depth(i, 0);
      }
    }
  }
  if (cv::sum(left_val)[0] < 500){
    std::cout << "number of valid after optimization is too small: " << cv::sum(left_val)[0] << std::endl;
    return -1;
  } else{
    return 0;
  }
}

GlobalStatus DepthEstimator::ComputeResidualJacobian(const Eigen::Matrix<float, Eigen::Dynamic, 1>& tmp_depth,
                                                     const std::vector<std::pair<int, int>>& coord_vec,
        Eigen::DiagonalMatrix<float, Eigen::Dynamic>& jtwj, Eigen::Matrix<float, Eigen::Dynamic, 1>& b,
                                     float& err_now, const int num_residuals, const cv::Mat& left_img, const cv::Mat& right_img,
                                                     Eigen::Matrix<float, Eigen::Dynamic, 1>& residuals){
  err_now = 0;
  int warped_x = 0;
  int num_residual_actual = 0; // used to compute averaged photometric error
  float r_i = 0;
  float r_i_diff = 0;
  float w_i = 0;
  float tx = baseline_; // in meters
  // TODO: only for debug now
  // float fx = camera_ptr_left_->fx_float(0); // in pixels
  float fx = 718.856f;
  //std::cout << "computing jacobian..." << std::endl;
  for (int i = 0; i < num_residuals; i++){
    warped_x = int(std::floor(coord_vec[i].first - tx * fx * tmp_depth(i)));
    // if warped out of boundary, leave some space to compute gradient
    if (warped_x < 2 || warped_x > left_img.cols-2){
      jtwj.diagonal()(i) = 0;
      b.row(i) << 0;
      residuals.row(i) << -1000;
      continue;
    }
    //std::cout << "i: " << i << ", before r_ri" << std::endl;
    r_i = left_img.at<float>(coord_vec[i].second, coord_vec[i].first) - right_img.at<float>(coord_vec[i].second, warped_x);
    //std::cout << "i: " << i << ", get r_ri" << std::endl;
    w_i = (std::fabs(r_i) <= huber_delta_) ? 1.0f : huber_delta_ / std::fabs(r_i);
    r_i_diff = tx * fx * 0.5f * (right_img.at<float>(coord_vec[i].second, warped_x+1) - right_img.at<float>(coord_vec[i].second, warped_x-1));
    //std::cout << "get r_i_diff" << std::endl;
    residuals.row(i) << std::fabs(r_i);
    num_residual_actual++;
    err_now += r_i * r_i * w_i;
    jtwj.diagonal()(i) = r_i_diff * r_i_diff * w_i;
    b.row(i) << - r_i_diff * w_i * r_i;
    //std::cout << "done" << std::endl;
  }

  err_now = (float(1.0) / float(num_residual_actual)) * err_now; // weighted error
  //std::cout << "err: " << err_now << " over " << num_residual_actual << " residuals." << std::endl;
  return 0;
}

GlobalStatus DepthEstimator::DisparityDepthEstimate(const cv::Mat& kleft_rect, const cv::Mat& kright_rect,
                                                     cv::Mat& left_disp, cv::Mat& left_dep, cv::Mat& left_val){

  /******** idea: set boundary start, end; loop over all pixels on the left rectified image ********/
  // a) compute grad on the pixel, if smaller than some TH continue, else
  // b) get the right boundary of epl
  // c) compute SSD value along the epl on the right rectified image, keep the smallest in a buffer
  // d) if the smallest SSD error is larger than some TH, return failed match; otherwise, set valid bool, store the value

  // check the memory
  // TODO: smooth left and right images
  cv::Mat left_rect, right_rect;
  cv::GaussianBlur(kleft_rect, left_rect, cv::Size(3, 3), 0);
  cv::GaussianBlur(kright_rect, right_rect, cv::Size(3, 3), 0);

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

  // get camera params
  // TODO: only for debug now
  //float fx = camera_ptr_left_->fx_float(0); // in pixels
  float fx = 718.856f;

  float current_ssd = 0;
  float smallest_ssd = 1e+10; // initial smallest ssd err
  int match_coord = -1; // the current best match column coord
  int begin_x = boundary_; // skip the first boundary_ cols of the image
  int end_x = left_rect.cols - boundary_; // skp the last boundary_ cols of the image, 640-boundary_
  int begin_y = boundary_; // skip the first boundary_ rows of the image
  int end_y = left_rect.rows - boundary_; // skp the last boundary_ rows of the image, 480-boundary_
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
  float* left_dep_row_ptr = nullptr;
  float grad_x = 0;
  float grad_y = 0;
  float mag_grad = 0;


  // KITTI size: 376x1241, 16x32 blocks
  cv::Mat grad_map(left_rect.rows, left_rect.cols, odometry::PixelType);
  int num_blocks = 16 * 32;
  int block_w = (left_rect.cols - boundary_ * 2) / 32;
  int block_h = (left_rect.rows - boundary_ * 2) / 16;
  std::vector<float> block_grad(block_w*block_h);
  int grad_count;
  int valid_count;
  float block_th;
  int start_y, start_x;
  for (int block_id = 0; block_id < num_blocks; block_id++){
    //std::cout << "block: " << block_id << std::endl;
    start_y = boundary_ + (block_id / 32) * block_h;
    start_x = boundary_ + (block_id % 32) * block_w;
    grad_count = 0;
    for (int y = start_y; y < start_y + block_h; y++){
      //std::cout << "y: " << y << std::endl;
      for (int x = start_x; x < start_x + block_w; x++){
        // compute gradient and store them to grad_map and block_grad vector
        grad_x = 0.5f * (left_rect.at<float>(y, x+1) - left_rect.at<float>(y, x-1));
        grad_y = 0.5f * (left_rect.at<float>(y+1, x) - left_rect.at<float>(y-1, x));
        mag_grad = std::sqrt(grad_x*grad_x + grad_y*grad_y);
        grad_map.at<float>(y, x) = mag_grad;
        block_grad[grad_count] = mag_grad;
        grad_count++;
      }
    }
    // compute median grad
    std::nth_element(block_grad.begin(), block_grad.begin() + block_grad.size()/2, block_grad.end());
    block_th = block_grad[block_grad.size()/2] + grad_th_;
    // select all points that have gradient larger than block_th
    valid_count = 0;
    for (int y = start_y; y < start_y + block_h; y++){
      for (int x = start_x; x < start_x + block_w; x++){
        if (valid_count >= 80) break;
        if (grad_map.at<float>(y, x) > block_th){
          left_val.at<uint8_t>(y, x) = 1;
          valid_count++;
        }
      }
      if (valid_count >= 80) break;
    }
  }


  __m256 left_pattern;
  for (int y=begin_y; y<end_y; y++){
    // get pointers
    left_p_row_ptr = left_rect.ptr<float>(y-1);
    left_row_ptr = left_rect.ptr<float>(y);
    left_n_row_ptr = left_rect.ptr<float>(y+1);
    left_val_row_ptr = left_val.ptr<uint8_t>(y);
    for (int x=begin_x; x<end_x; x++){
      // check if a valid point
      if (*(left_val_row_ptr+x) == 0) continue;
      else {
        // get pointers for left_disp, left_dep
        left_disp_row_ptr = left_disp.ptr<float>(y);
        left_pp_row_ptr = left_rect.ptr<float>(y-2);
        left_nn_row_ptr = left_rect.ptr<float>(y+2);
        left_dep_row_ptr = left_dep.ptr<float>(y);
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
          current_ssd = ComputeSsdPattern8(left_pp_row_ptr, left_p_row_ptr, left_row_ptr, left_n_row_ptr, left_nn_row_ptr,
                                      right_pp_row_ptr, right_p_row_ptr, right_row_ptr, right_n_row_ptr, right_nn_row_ptr, x, right_x);
          match_coord = (current_ssd < smallest_ssd) ? (right_x) : match_coord;
          smallest_ssd = (current_ssd < smallest_ssd) ? (current_ssd) : smallest_ssd;
        } // loop right cols
        */
        /***************** Search along epl: SSE/AVX implementation **************/
        left_pattern = _mm256_set_ps(*(left_pp_row_ptr+x), *(left_p_row_ptr+x-1), *(left_p_row_ptr+x+1), *(left_row_ptr+x-2),
                                     *(left_row_ptr+x), *(left_row_ptr+x+2), *(left_n_row_ptr+x-1), *(left_nn_row_ptr+x));
        for (int right_x=begin_x; right_x<x; right_x++){
          ComputeSsdPattern8Sse(left_pattern, right_pp_row_ptr, right_p_row_ptr, right_row_ptr, right_n_row_ptr,
                  right_nn_row_ptr, right_x, &current_ssd);
          match_coord = (current_ssd < smallest_ssd) ? (right_x) : match_coord;
          smallest_ssd = (current_ssd < smallest_ssd) ? (current_ssd) : smallest_ssd;
        } // loop right cols
        if (smallest_ssd > ssd_th_)
          continue;
        else {
          *(left_disp_row_ptr+x) = std::abs(x-match_coord); // left_disp.at<float>(y, x) = std::abs(x-match_coord);
          // compute left inverse depth value using rectified Camera baseline and Intrinsic:
          // depth = fx * baseline / disp, fx: [pixels], baseline: [meters], disp: [pixels]
          *(left_dep_row_ptr+x) = *(left_disp_row_ptr+x) / (fx * baseline_);
        } // a successful match, store the disparity value, set valid mask
      } // if left grad is large
    } // loop left cols
  } // loop left rows

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

inline float DepthEstimator::ComputeSsdPattern8(const float* left_pp_row_ptr, const float* left_p_row_ptr, const float* left_row_ptr, const float* left_n_row_ptr, const float* left_nn_row_ptr,
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

inline void DepthEstimator::ComputeSsdPattern8Sse(const __m256& left_pattern, const float* right_pp_row_ptr, const float* right_p_row_ptr,
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

void DepthEstimator::ReportStatus(){
  std::cout << "    Number of iters performed: " << iters_stat_ << "(max allowed: " << max_iters_ << ")" << std::endl;
  std::cout << "    Final cost: " << cost_stat_ << std::endl;
}

} // namespace odometry

