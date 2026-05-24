#ifndef GRAD_SCALER_HPP
#define GRAD_SCALER_HPP

#include "core/tensor.hpp"
#include "optim/optimizer.hpp"
#include <memory>

class AMPContext {
public:
  [[nodiscard]] static bool is_enabled() noexcept;
  static void set_enabled(bool enabled) noexcept;

  [[nodiscard]] static DataType active_dtype() noexcept;
  static void set_active_dtype(DataType dtype) noexcept;

private:
  static thread_local bool enabled_;
  static thread_local DataType active_dtype_;
};

class AMPGuard {
public:
  explicit AMPGuard(bool enabled = true, DataType dtype = DataType::FP16) noexcept;
  ~AMPGuard();

  AMPGuard(const AMPGuard &) = delete;
  AMPGuard &operator=(const AMPGuard &) = delete;
  AMPGuard(AMPGuard &&) noexcept = default;
  AMPGuard &operator=(AMPGuard &&) noexcept = default;

private:
  bool prev_enabled_;
  DataType prev_dtype_;
};

class GradScaler {
public:
  explicit GradScaler(float init_scale = 65536.0f, float growth_factor = 2.0f,
                      float backoff_factor = 0.5f, int growth_interval = 2000);

  [[nodiscard]] std::shared_ptr<Tensor> scale(const std::shared_ptr<Tensor> &loss);
  void unscale(Optimizer &optimizer);
  bool step(Optimizer &optimizer);
  void update();

  [[nodiscard]] float scale_factor() const noexcept { return scale_factor_; }

private:
  float scale_factor_;
  float growth_factor_;
  float backoff_factor_;
  int growth_interval_;
  int growth_tracker_ = 0;
  bool unscaled_ = false;
  bool has_overflow_ = false;
};

#endif // GRAD_SCALER_HPP
