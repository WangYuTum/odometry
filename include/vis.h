
#ifndef VIS_H
#define VIS_H

#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <opencv2/core/core.hpp>
#include <nanogui/nanogui.h>
#include <nanogui/imagepanel.h>
#include <nanogui/imageview.h>

#include "keyframe.h"
#include "trajectory_view.h"
#include "vis_screen.h"

class Vis {
public:
  Vis();
  void start();
  void loadNewestKeyframe(const odometry::KeyFrame &);

private:
  double m_lastFrameTime;
  int m_numElapsedFrames;
  double m_fps;
  GLuint m_rgbLeftTexId;
  GLuint m_rgbRightTexId;
  GLuint m_depthLeftTexId;

  std::vector<odometry::KeyFrame> m_keyframeBuffer;
  TrajectoryView *m_view;
};

#endif