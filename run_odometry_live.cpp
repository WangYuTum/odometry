// The file runs full pipline of odometry on live stereo camera.
// Camera parameters are read from calibration file.
// Multi-thread is used to guarantee real-time.
// Created by Yu Wang on 2019-01-13.

// Note:
// Camera Output:
//  * MUST be created with dynamic allocator (shared pointer)
//  * MUST be aligned to 32-bit address

int main(){

  /********************************* System initialisation ************************************/

  // create/setup stereo camera instance: call SetUpStereoCameraSystem()
  //  * this will create left/right camera pyramid with rectified intrinsics
  //  * this will also create valid regions, which will be needed later
  //  * intersect valid region with pre-defined valid boundary, multiple of 4

  // create camera output buffer

  // create GUI (nanogui, and all other necessary windows)

  /********************************* Tracking ************************************/

  // Compute depth (need valid region)
  //  * output valid map, which will be used by tracking (do not need valid region anymore)

  // Compute pose (need valid map)

  return 0;
}