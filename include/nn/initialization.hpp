#ifndef INITIALIZATION_HPP
#define INITIALIZATION_HPP

#include "core/tensor.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

namespace init {

inline void constant(Tensor &tensor, float val) { tensor.fill(val); }

inline void uniform(Tensor &tensor, float low, float high) {
  size_t size = count_elements(tensor.shape());
  std::vector<float> data(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(low, high);
  for (size_t i = 0; i < size; ++i) {
    data[i] = dis(gen);
  }
  tensor.copy_from_host(data);
}

inline void normal(Tensor &tensor, float mean, float stddev) {
  size_t size = count_elements(tensor.shape());
  std::vector<float> data(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dis(mean, stddev);
  for (size_t i = 0; i < size; ++i) {
    data[i] = dis(gen);
  }
  tensor.copy_from_host(data);
}

inline void xavier_uniform(Tensor &tensor) {
  const auto &shape = tensor.shape();
  if (shape.size() < 2) {
    throw std::runtime_error(
        "Xavier uniform initialization requires at least a 2D tensor.");
  }
  size_t fan_in = shape[1];
  size_t fan_out = shape[0];
  for (size_t i = 2; i < shape.size(); ++i) {
    fan_in *= shape[i];
    fan_out *= shape[i];
  }
  float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
  uniform(tensor, -limit, limit);
}

inline void kaiming_uniform(Tensor &tensor) {
  const auto &shape = tensor.shape();
  if (shape.size() < 2) {
    throw std::runtime_error(
        "Kaiming uniform initialization requires at least a 2D tensor.");
  }
  size_t fan_in = shape[1];
  for (size_t i = 2; i < shape.size(); ++i) {
    fan_in *= shape[i];
  }
  float bound = std::sqrt(3.0f / static_cast<float>(fan_in));
  uniform(tensor, -bound, bound);
}

} // namespace init

#endif // INITIALIZATION_HPP
