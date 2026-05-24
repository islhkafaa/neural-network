#include "optim/lr_scheduler.hpp"
#include <algorithm>
#include <cmath>

LRScheduler::LRScheduler(Optimizer &optimizer)
    : optimizer_(optimizer), initial_lr_(optimizer.lr()) {}

StepLR::StepLR(Optimizer &optimizer, size_t step_size, float gamma)
    : LRScheduler(optimizer), step_size_(step_size), gamma_(gamma) {}

void StepLR::step() {
  last_epoch_++;
  size_t decay_steps = last_epoch_ / step_size_;
  float new_lr =
      initial_lr_ * std::pow(gamma_, static_cast<float>(decay_steps));
  optimizer_.set_lr(new_lr);
}

CosineAnnealingLR::CosineAnnealingLR(Optimizer &optimizer, size_t T_max,
                                     float eta_min)
    : LRScheduler(optimizer), T_max_(T_max), eta_min_(eta_min) {}

void CosineAnnealingLR::step() {
  last_epoch_++;
  size_t T_cur = std::min(last_epoch_, T_max_);
  const float pi = 3.1415926535f;
  float new_lr =
      eta_min_ + 0.5f * (initial_lr_ - eta_min_) *
                     (1.0f + std::cos(static_cast<float>(T_cur) / T_max_ * pi));
  optimizer_.set_lr(new_lr);
}
