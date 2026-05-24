#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "backend/backend.hpp"
#include "backend/cpu_backend.hpp"
#include "core/shape.hpp"
#include <functional>
#include <memory>
#include <vector>

inline ExecutionBackend *default_backend() {
  static CpuBackend instance;
  return &instance;
}

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
  explicit Tensor(const Shape &shape, ExecutionBackend *backend = nullptr);
  Tensor(const Shape &shape, const std::vector<float> &host_data,
         ExecutionBackend *backend = nullptr);
  Tensor(const Shape &shape, const Strides &strides, size_t offset,
         std::shared_ptr<DeviceBuffer> storage,
         ExecutionBackend *backend = nullptr);

  [[nodiscard]] const Shape &shape() const noexcept { return shape_; }
  [[nodiscard]] const Strides &strides() const noexcept { return strides_; }
  [[nodiscard]] size_t offset() const noexcept { return offset_; }
  [[nodiscard]] std::shared_ptr<DeviceBuffer> storage() const noexcept {
    return storage_;
  }
  [[nodiscard]] ExecutionBackend *backend() const noexcept { return backend_; }

  [[nodiscard]] bool requires_grad() const noexcept { return requires_grad_; }
  void set_requires_grad(bool req) noexcept { requires_grad_ = req; }

  [[nodiscard]] std::shared_ptr<Tensor> grad() const noexcept { return grad_; }
  void set_grad(std::shared_ptr<Tensor> g) noexcept { grad_ = g; }

  [[nodiscard]] const std::vector<std::shared_ptr<Tensor>> &
  inputs() const noexcept {
    return inputs_;
  }

  [[nodiscard]] std::vector<float> to_host() const;
  void fill(float value);
  void copy_from_host(const std::vector<float> &host_data);

  std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor> &other);
  std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor> &other);
  std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor> &other);
  std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor> &other);
  std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor> &other);
  std::shared_ptr<Tensor> sum(const std::vector<size_t> &axes,
                              bool keep_dims = false);
  std::shared_ptr<Tensor> transpose();
  std::shared_ptr<Tensor> transpose(size_t dim1, size_t dim2);
  std::shared_ptr<Tensor> reshape(const Shape &new_shape);
  std::shared_ptr<Tensor> bmm(const std::shared_ptr<Tensor> &other);
  std::shared_ptr<Tensor> relu();
  std::shared_ptr<Tensor> sigmoid();
  std::shared_ptr<Tensor> tanh();
  std::shared_ptr<Tensor> softmax(size_t axis);
  std::shared_ptr<Tensor> sqrt();
  std::shared_ptr<Tensor> log();
  std::shared_ptr<Tensor> conv2d(const std::shared_ptr<Tensor> &weight,
                                 const std::shared_ptr<Tensor> &bias,
                                 size_t padding = 0, size_t stride = 1);
  std::shared_ptr<Tensor> maxpool2d(size_t pool_h, size_t pool_w,
                                    size_t stride = 1);
  std::shared_ptr<Tensor> embedding(const std::shared_ptr<Tensor> &weight);
  std::shared_ptr<Tensor> slice(size_t axis, size_t index);
  std::shared_ptr<Tensor> rope(size_t S_past);
  static std::shared_ptr<Tensor>
  stack(const std::vector<std::shared_ptr<Tensor>> &tensors);

  void backward();
  void zero_grad() noexcept;
  void accumulate_grad(const std::shared_ptr<Tensor> &incoming_grad);

private:
  Shape shape_;
  Strides strides_;
  size_t offset_ = 0;
  std::shared_ptr<DeviceBuffer> storage_;
  ExecutionBackend *backend_;

  bool requires_grad_ = false;
  std::shared_ptr<Tensor> grad_;
  std::vector<std::shared_ptr<Tensor>> inputs_;
  std::function<void()> backward_;

  friend std::shared_ptr<Tensor>
  reduce_grad_for_broadcasting(const std::shared_ptr<Tensor> &grad,
                               const Shape &target_shape);
};

inline std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &lhs,
                                         const std::shared_ptr<Tensor> &rhs) {
  return lhs->add(rhs);
}

inline std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor> &lhs,
                                         const std::shared_ptr<Tensor> &rhs) {
  return lhs->sub(rhs);
}

inline std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &lhs,
                                         const std::shared_ptr<Tensor> &rhs) {
  return lhs->mul(rhs);
}

inline std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor> &lhs,
                                         const std::shared_ptr<Tensor> &rhs) {
  return lhs->div(rhs);
}

inline std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor> &lhs,
                                         float rhs) {
  auto scalar = std::make_shared<Tensor>(Shape{}, std::vector<float>{rhs},
                                         lhs->backend());
  return lhs->div(scalar);
}

inline std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &lhs,
                                         float rhs) {
  auto scalar = std::make_shared<Tensor>(Shape{}, std::vector<float>{rhs},
                                         lhs->backend());
  return lhs->mul(scalar);
}

inline std::shared_ptr<Tensor> operator*(float lhs,
                                         const std::shared_ptr<Tensor> &rhs) {
  auto scalar = std::make_shared<Tensor>(Shape{}, std::vector<float>{lhs},
                                         rhs->backend());
  return scalar->mul(rhs);
}

inline std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor> &val) {
  auto zero = std::make_shared<Tensor>(val->shape(), val->backend());
  zero->fill(0.0f);
  return zero->sub(val);
}

#endif // TENSOR_HPP
