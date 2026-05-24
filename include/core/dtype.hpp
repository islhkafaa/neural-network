#ifndef DTYPE_HPP
#define DTYPE_HPP

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>

enum class DataType {
  FP32,
  FP16,
  BF16
};

inline size_t dtype_size(DataType dtype) noexcept {
  switch (dtype) {
    case DataType::FP32: return 4;
    case DataType::FP16: return 2;
    case DataType::BF16: return 2;
  }
  return 4;
}

inline uint16_t float_to_fp16(float f) noexcept {
  uint32_t i;
  std::memcpy(&i, &f, sizeof(float));
  int s = (i >> 16) & 0x00008000;
  int e = ((i >> 23) & 0x000000ff) - (127 - 15);
  int m = i & 0x007fffff;
  if (e <= 0) {
    if (e < -10) return s;
    m = m | 0x00800000;
    int t = 14 - e;
    int a = (1 << (t - 1)) - 1;
    int b = (m >> t) & 1;
    m = (m + a + b) >> t;
    return s | m;
  } else if (e >= 31) {
    return s | 0x7c00;
  }
  int a = 0x00000fff + ((m >> 13) & 1);
  m = (m + a) >> 13;
  if (m & 0x00000400) {
    m &= ~0x00000400;
    e += 1;
  }
  if (e >= 31) return s | 0x7c00;
  return s | (e << 10) | m;
}

inline float fp16_to_float(uint16_t h) noexcept {
  int s = (h >> 15) & 0x00000001;
  int e = (h >> 10) & 0x0000001f;
  int m = h & 0x000003ff;
  uint32_t i;
  if (e == 0) {
    if (m == 0) {
      i = s << 31;
      float f;
      std::memcpy(&f, &i, sizeof(float));
      return f;
    }
    while (!(m & 0x00000400)) {
      m <<= 1;
      e -= 1;
    }
    e += 1;
    m &= ~0x00000400;
  } else if (e == 31) {
    i = (s << 31) | 0x7f800000 | (m << 13);
    float f;
    std::memcpy(&f, &i, sizeof(float));
    return f;
  }
  e = e + (127 - 15);
  m = m << 13;
  i = (s << 31) | (e << 23) | m;
  float f;
  std::memcpy(&f, &i, sizeof(float));
  return f;
}

inline uint16_t float_to_bfloat16(float f) noexcept {
  uint32_t val;
  std::memcpy(&val, &f, sizeof(float));
  uint32_t rounding_bias = 0x7fff + ((val >> 16) & 1);
  val += rounding_bias;
  return static_cast<uint16_t>(val >> 16);
}

inline float bfloat16_to_float(uint16_t b) noexcept {
  uint32_t val = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &val, sizeof(float));
  return f;
}

#endif // DTYPE_HPP
