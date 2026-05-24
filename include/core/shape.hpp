#ifndef SHAPE_HPP
#define SHAPE_HPP

#include <cstddef>
#include <numeric>
#include <vector>

using Shape = std::vector<size_t>;
using Strides = std::vector<size_t>;

inline size_t count_elements(const Shape &shape) {
  if (shape.empty()) {
    return 1;
  }
  return std::accumulate(shape.begin(), shape.end(), size_t{1},
                         std::multiplies<size_t>());
}

inline Strides compute_strides(const Shape &shape) {
  if (shape.empty()) {
    return {};
  }
  Strides strides(shape.size());
  size_t current_stride = 1;
  for (size_t i = shape.size(); i > 0; --i) {
    strides[i - 1] = current_stride;
    current_stride *= shape[i - 1];
  }
  return strides;
}

inline bool is_contiguous(const Shape &shape, const Strides &strides) {
  if (shape.empty()) {
    return true;
  }
  if (shape.size() != strides.size()) {
    return false;
  }
  size_t expected_stride = 1;
  for (size_t i = shape.size(); i > 0; --i) {
    if (strides[i - 1] != expected_stride) {
      return false;
    }
    expected_stride *= shape[i - 1];
  }
  return true;
}

#endif // SHAPE_HPP
