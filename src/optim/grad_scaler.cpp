#include "optim/grad_scaler.hpp"
#include <cmath>

thread_local bool AMPContext::enabled_ = false;
thread_local DataType AMPContext::active_dtype_ = DataType::FP16;

bool AMPContext::is_enabled() noexcept { return enabled_; }
void AMPContext::set_enabled(bool enabled) noexcept { enabled_ = enabled; }

DataType AMPContext::active_dtype() noexcept { return active_dtype_; }
void AMPContext::set_active_dtype(DataType dtype) noexcept { active_dtype_ = dtype; }

AMPGuard::AMPGuard(bool enabled, DataType dtype) noexcept {
  prev_enabled_ = AMPContext::is_enabled();
  prev_dtype_ = AMPContext::active_dtype();
  AMPContext::set_enabled(enabled);
  AMPContext::set_active_dtype(dtype);
}

AMPGuard::~AMPGuard() {
  AMPContext::set_enabled(prev_enabled_);
  AMPContext::set_active_dtype(prev_dtype_);
}

GradScaler::GradScaler(float init_scale, float growth_factor,
                       float backoff_factor, int growth_interval)
    : scale_factor_(init_scale), growth_factor_(growth_factor),
      backoff_factor_(backoff_factor), growth_interval_(growth_interval) {}

std::shared_ptr<Tensor> GradScaler::scale(const std::shared_ptr<Tensor> &loss) {
  return loss * scale_factor_;
}

void GradScaler::unscale(Optimizer &optimizer) {
  if (unscaled_) return;

  has_overflow_ = false;
  float inv_scale = 1.0f / scale_factor_;

  for (auto &param : optimizer.params()) {
    if (param->requires_grad() && param->grad()) {
      auto g = param->grad();
      std::vector<float> g_data = g->to_host();
      for (size_t i = 0; i < g_data.size(); ++i) {
        float val = g_data[i];
        if (std::isnan(val) || std::isinf(val)) {
          has_overflow_ = true;
        }
        g_data[i] = val * inv_scale;
      }
      g->copy_from_host(g_data);
    }
  }
  unscaled_ = true;
}

bool GradScaler::step(Optimizer &optimizer) {
  unscale(optimizer);
  if (has_overflow_) {
    return false;
  }
  optimizer.step();
  return true;
}

void GradScaler::update() {
  if (has_overflow_) {
    scale_factor_ *= backoff_factor_;
    growth_tracker_ = 0;
  } else {
    growth_tracker_++;
    if (growth_tracker_ >= growth_interval_) {
      scale_factor_ *= growth_factor_;
      growth_tracker_ = 0;
    }
  }
  unscaled_ = false;
  has_overflow_ = false;
}
