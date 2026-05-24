#ifndef LR_SCHEDULER_HPP
#define LR_SCHEDULER_HPP

#include "optim/optimizer.hpp"

class LRScheduler {
public:
  explicit LRScheduler(Optimizer &optimizer);
  virtual ~LRScheduler() = default;

  virtual void step() = 0;

protected:
  Optimizer &optimizer_;
  float initial_lr_;
  size_t last_epoch_ = 0;
};

class StepLR : public LRScheduler {
public:
  StepLR(Optimizer &optimizer, size_t step_size, float gamma = 0.1f);

  void step() override;

private:
  size_t step_size_;
  float gamma_;
};

class CosineAnnealingLR : public LRScheduler {
public:
  CosineAnnealingLR(Optimizer &optimizer, size_t T_max, float eta_min = 0.0f);

  void step() override;

private:
  size_t T_max_;
  float eta_min_;
};

#endif // LR_SCHEDULER_HPP
