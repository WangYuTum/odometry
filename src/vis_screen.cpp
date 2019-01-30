// Created by Max on 28.12.18.
//

#include <vis_screen.h>

// TODO: Add possibility to register callback
// The callback will be put in Vis.cpp and does the actual
// data reloading.

void VisScreen::onUpdate(const std::function<void()> &callback) {
  m_update_callback = callback;
}