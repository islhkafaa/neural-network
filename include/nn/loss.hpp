#ifndef LOSS_HPP
#define LOSS_HPP

#include "core/tensor.hpp"
#include <memory>

class Loss {
public:
  virtual ~Loss() = default;
  virtual std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &pred,
          const std::shared_ptr<Tensor> &target) = 0;
};

class MSELoss : public Loss {
public:
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &pred,
          const std::shared_ptr<Tensor> &target) override;
};

class BCELoss : public Loss {
public:
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &pred,
          const std::shared_ptr<Tensor> &target) override;
};

class SoftmaxCrossEntropyLoss : public Loss {
public:
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &pred,
          const std::shared_ptr<Tensor> &target) override;
};

#endif // LOSS_HPP
