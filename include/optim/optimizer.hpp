#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "core/tensor.hpp"
#include <memory>
#include <vector>

class Optimizer {
public:
  explicit Optimizer(std::vector<std::shared_ptr<Tensor>> params,
                     float lr = 1e-3f, float weight_decay = 0.0f);
  virtual ~Optimizer() = default;

  virtual void step() = 0;
  void zero_grad() noexcept;

  void clip_grad_norm(float max_norm);
  void clip_grad_value(float clip_value);

  [[nodiscard]] float lr() const noexcept { return lr_; }
  void set_lr(float lr) noexcept { lr_ = lr; }

protected:
  std::vector<std::shared_ptr<Tensor>> params_;
  float lr_;
  float weight_decay_;
};

class SGD : public Optimizer {
public:
  SGD(std::vector<std::shared_ptr<Tensor>> params, float lr = 1e-3f,
      float momentum = 0.0f, float weight_decay = 0.0f, bool nesterov = false);

  void step() override;

private:
  float momentum_;
  bool nesterov_;
  std::vector<std::vector<float>> velocities_;
};

class Adam : public Optimizer {
public:
  Adam(std::vector<std::shared_ptr<Tensor>> params, float lr = 1e-3f,
       float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
       float weight_decay = 0.0f);

  void step() override;

private:
  float beta1_;
  float beta2_;
  float eps_;
  size_t t_ = 0;
  std::vector<std::vector<float>> m_;
  std::vector<std::vector<float>> v_;
};

class RMSprop : public Optimizer {
public:
  RMSprop(std::vector<std::shared_ptr<Tensor>> params, float lr = 1e-2f,
          float alpha = 0.99f, float eps = 1e-8f, float weight_decay = 0.0f,
          float momentum = 0.0f);

  void step() override;

private:
  float alpha_;
  float eps_;
  float momentum_;
  std::vector<std::vector<float>> square_avgs_;
  std::vector<std::vector<float>> velocities_;
};

class AdamW : public Optimizer {
public:
  AdamW(std::vector<std::shared_ptr<Tensor>> params, float lr = 1e-3f,
        float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
        float weight_decay = 0.01f);

  void step() override;

private:
  float beta1_;
  float beta2_;
  float eps_;
  size_t t_ = 0;
  std::vector<std::vector<float>> m_;
  std::vector<std::vector<float>> v_;
};

class Adafactor : public Optimizer {
public:
  Adafactor(std::vector<std::shared_ptr<Tensor>> params, float lr = 1e-3f,
            float beta2 = 0.999f, float eps1 = 1e-30f, float eps2 = 1e-3f,
            float weight_decay = 0.0f);

  void step() override;

private:
  float beta2_;
  float eps1_;
  float eps2_;
  std::vector<std::vector<float>> r_;
  std::vector<std::vector<float>> c_;
  std::vector<std::vector<float>> v_;
};

#endif // OPTIMIZER_HPP
