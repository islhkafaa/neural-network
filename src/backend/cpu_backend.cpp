#include "backend/cpu_backend.hpp"
#include "backend/thread_pool.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>

extern "C" {
void cblas_sgemm(const int Order, const int TransA, const int TransB,
                 const int M, const int N, const int K, const float alpha,
                 const float *A, const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);
}

constexpr int CblasRowMajor = 101;
constexpr int CblasNoTrans = 111;
constexpr int CblasTrans = 112;

CpuBuffer::CpuBuffer(size_t bytes) : data_(bytes, 0) {}

size_t CpuBuffer::size() const noexcept { return data_.size(); }

void *CpuBuffer::data() noexcept { return data_.data(); }

const void *CpuBuffer::data() const noexcept { return data_.data(); }

std::unique_ptr<DeviceBuffer> CpuBackend::allocate(size_t bytes) {
  return std::make_unique<CpuBuffer>(bytes);
}

void CpuBackend::copy_host_to_device(const float *host_ptr,
                                     DeviceBuffer &device_buf,
                                     size_t size_bytes) {
  std::memcpy(device_buf.data(), host_ptr, size_bytes);
}

void CpuBackend::copy_to_device(const float *host_ptr, DeviceBuffer &device_buf,
                                size_t dest_offset_bytes, size_t size_bytes) {
  std::memcpy(static_cast<char *>(device_buf.data()) + dest_offset_bytes,
              host_ptr, size_bytes);
}

void CpuBackend::copy_device_to_host(const DeviceBuffer &device_buf,
                                     float *host_ptr, size_t size_bytes) {
  std::memcpy(host_ptr, device_buf.data(), size_bytes);
}

void CpuBackend::fill(DeviceBuffer &buffer, float value) {
  if (value == 0.0f) {
    std::memset(buffer.data(), 0, buffer.size());
    return;
  }
  DataType dt = buffer.dtype();
  size_t elements = buffer.size() / dtype_size(dt);
  if (dt == DataType::FP32) {
    float *ptr = static_cast<float *>(buffer.data());
    std::fill(ptr, ptr + elements, value);
  } else if (dt == DataType::FP16) {
    uint16_t *ptr = static_cast<uint16_t *>(buffer.data());
    uint16_t half_val = float_to_fp16(value);
    std::fill(ptr, ptr + elements, half_val);
  } else {
    uint16_t *ptr = static_cast<uint16_t *>(buffer.data());
    uint16_t half_val = float_to_bfloat16(value);
    std::fill(ptr, ptr + elements, half_val);
  }
}

static size_t get_input_offset(size_t flat_out_idx, const Shape &out_shape,
                               const Shape &in_shape,
                               const Strides &in_strides) {
  size_t R = out_shape.size();
  size_t R_in = in_shape.size();
  size_t temp = flat_out_idx;
  size_t in_offset = 0;

  for (size_t i = 0; i < R; ++i) {
    size_t dim = R - 1 - i;
    size_t coord = temp % out_shape[dim];
    temp /= out_shape[dim];

    if (dim >= R - R_in) {
      size_t in_dim = dim - (R - R_in);
      if (in_shape[in_dim] > 1) {
        in_offset += coord * in_strides[in_dim];
      }
    }
  }
  return in_offset;
}

static float read_value(const void *data, size_t index, DataType dtype) noexcept {
  if (dtype == DataType::FP32) {
    return static_cast<const float *>(data)[index];
  } else if (dtype == DataType::FP16) {
    return fp16_to_float(static_cast<const uint16_t *>(data)[index]);
  } else {
    return bfloat16_to_float(static_cast<const uint16_t *>(data)[index]);
  }
}

static void write_value(void *data, size_t index, float val, DataType dtype) noexcept {
  if (dtype == DataType::FP32) {
    static_cast<float *>(data)[index] = val;
  } else if (dtype == DataType::FP16) {
    static_cast<uint16_t *>(data)[index] = float_to_fp16(val);
  } else {
    static_cast<uint16_t *>(data)[index] = float_to_bfloat16(val);
  }
}

template <typename Op>
static void
elementwise_op(const DeviceBuffer &a, size_t a_offset, const Shape &a_shape,
               const Strides &a_strides, const DeviceBuffer &b, size_t b_offset,
               const Shape &b_shape, const Strides &b_strides,
               DeviceBuffer &out, size_t out_offset, const Shape &out_shape,
               const Strides &out_strides, Op op) {
  const void *a_data = a.data();
  const void *b_data = b.data();
  void *out_data = out.data();

  DataType a_dt = a.dtype();
  DataType b_dt = b.dtype();
  DataType out_dt = out.dtype();

  size_t out_elements = count_elements(out_shape);

  if (is_contiguous(a_shape, a_strides) && is_contiguous(b_shape, b_strides) &&
      is_contiguous(out_shape, out_strides) && a_shape == b_shape) {
    if (out_elements >= 2048) {
      ThreadPool::instance().parallel_for(0, out_elements, [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
          float val_a = read_value(a_data, a_offset + i, a_dt);
          float val_b = read_value(b_data, b_offset + i, b_dt);
          write_value(out_data, out_offset + i, op(val_a, val_b), out_dt);
        }
      });
    } else {
      for (size_t i = 0; i < out_elements; ++i) {
        float val_a = read_value(a_data, a_offset + i, a_dt);
        float val_b = read_value(b_data, b_offset + i, b_dt);
        write_value(out_data, out_offset + i, op(val_a, val_b), out_dt);
      }
    }
    return;
  }

  if (out_elements >= 2048) {
    ThreadPool::instance().parallel_for(0, out_elements, [&](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        size_t a_idx = get_input_offset(i, out_shape, a_shape, a_strides);
        size_t b_idx = get_input_offset(i, out_shape, b_shape, b_strides);
        size_t out_idx = get_input_offset(i, out_shape, out_shape, out_strides);
        float val_a = read_value(a_data, a_offset + a_idx, a_dt);
        float val_b = read_value(b_data, b_offset + b_idx, b_dt);
        write_value(out_data, out_offset + out_idx, op(val_a, val_b), out_dt);
      }
    });
  } else {
    for (size_t i = 0; i < out_elements; ++i) {
      size_t a_idx = get_input_offset(i, out_shape, a_shape, a_strides);
      size_t b_idx = get_input_offset(i, out_shape, b_shape, b_strides);
      size_t out_idx = get_input_offset(i, out_shape, out_shape, out_strides);
      float val_a = read_value(a_data, a_offset + a_idx, a_dt);
      float val_b = read_value(b_data, b_offset + b_idx, b_dt);
      write_value(out_data, out_offset + out_idx, op(val_a, val_b), out_dt);
    }
  }
}

template <typename Op>
static void
unary_op(const DeviceBuffer &in, size_t in_offset, const Shape &in_shape,
         const Strides &in_strides, DeviceBuffer &out, size_t out_offset,
         const Shape &out_shape, const Strides &out_strides, Op op) {
  const void *in_data = in.data();
  void *out_data = out.data();
  DataType in_dt = in.dtype();
  DataType out_dt = out.dtype();
  size_t out_elements = count_elements(out_shape);

  if (is_contiguous(in_shape, in_strides) && is_contiguous(out_shape, out_strides)) {
    if (out_elements >= 2048) {
      ThreadPool::instance().parallel_for(0, out_elements, [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
          float val_in = read_value(in_data, in_offset + i, in_dt);
          write_value(out_data, out_offset + i, op(val_in), out_dt);
        }
      });
    } else {
      for (size_t i = 0; i < out_elements; ++i) {
        float val_in = read_value(in_data, in_offset + i, in_dt);
        write_value(out_data, out_offset + i, op(val_in), out_dt);
      }
    }
    return;
  }

  if (out_elements >= 2048) {
    ThreadPool::instance().parallel_for(0, out_elements, [&](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        size_t in_idx = get_input_offset(i, out_shape, in_shape, in_strides);
        size_t out_idx = get_input_offset(i, out_shape, out_shape, out_strides);
        float val_in = read_value(in_data, in_offset + in_idx, in_dt);
        write_value(out_data, out_offset + out_idx, op(val_in), out_dt);
      }
    });
  } else {
    for (size_t i = 0; i < out_elements; ++i) {
      size_t in_idx = get_input_offset(i, out_shape, in_shape, in_strides);
      size_t out_idx = get_input_offset(i, out_shape, out_shape, out_strides);
      float val_in = read_value(in_data, in_offset + in_idx, in_dt);
      write_value(out_data, out_offset + out_idx, op(val_in), out_dt);
    }
  }
}

template <typename Op>
static void
binary_backward_op(const DeviceBuffer &in, size_t in_offset, const Shape &in_shape,
                   const Strides &in_strides, const DeviceBuffer &grad_out,
                   size_t grad_out_offset, const Shape &grad_out_shape,
                   const Strides &grad_out_strides, DeviceBuffer &grad_in,
                   size_t grad_in_offset, const Shape &grad_in_shape,
                   const Strides &grad_in_strides, Op op) {
  const void *in_data = in.data();
  const void *go_data = grad_out.data();
  void *gi_data = grad_in.data();

  DataType in_dt = in.dtype();
  DataType go_dt = grad_out.dtype();
  DataType gi_dt = grad_in.dtype();

  size_t in_elements = count_elements(in_shape);

  if (is_contiguous(in_shape, in_strides) &&
      is_contiguous(grad_out_shape, grad_out_strides) &&
      is_contiguous(grad_in_shape, grad_in_strides)) {
    if (in_elements >= 2048) {
      ThreadPool::instance().parallel_for(0, in_elements, [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
          float val_in = read_value(in_data, in_offset + i, in_dt);
          float val_go = read_value(go_data, grad_out_offset + i, go_dt);
          write_value(gi_data, grad_in_offset + i, op(val_in, val_go), gi_dt);
        }
      });
    } else {
      for (size_t i = 0; i < in_elements; ++i) {
        float val_in = read_value(in_data, in_offset + i, in_dt);
        float val_go = read_value(go_data, grad_out_offset + i, go_dt);
        write_value(gi_data, grad_in_offset + i, op(val_in, val_go), gi_dt);
      }
    }
    return;
  }

  if (in_elements >= 2048) {
    ThreadPool::instance().parallel_for(0, in_elements, [&](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        size_t in_idx = get_input_offset(i, in_shape, in_shape, in_strides);
        size_t go_idx = get_input_offset(i, in_shape, grad_out_shape, grad_out_strides);
        size_t gi_idx = get_input_offset(i, in_shape, grad_in_shape, grad_in_strides);
        float val_in = read_value(in_data, in_offset + in_idx, in_dt);
        float val_go = read_value(go_data, grad_out_offset + go_idx, go_dt);
        write_value(gi_data, grad_in_offset + gi_idx, op(val_in, val_go), gi_dt);
      }
    });
  } else {
    for (size_t i = 0; i < in_elements; ++i) {
      size_t in_idx = get_input_offset(i, in_shape, in_shape, in_strides);
      size_t go_idx = get_input_offset(i, in_shape, grad_out_shape, grad_out_strides);
      size_t gi_idx = get_input_offset(i, in_shape, grad_in_shape, grad_in_strides);
      float val_in = read_value(in_data, in_offset + in_idx, in_dt);
      float val_go = read_value(go_data, grad_out_offset + go_idx, go_dt);
      write_value(gi_data, grad_in_offset + gi_idx, op(val_in, val_go), gi_dt);
    }
  }
}

void CpuBackend::add(const DeviceBuffer &a, size_t a_offset,
                     const Shape &a_shape, const Strides &a_strides,
                     const DeviceBuffer &b, size_t b_offset,
                     const Shape &b_shape, const Strides &b_strides,
                     DeviceBuffer &out, size_t out_offset,
                     const Shape &out_shape, const Strides &out_strides) {
  elementwise_op(a, a_offset, a_shape, a_strides, b, b_offset, b_shape,
                 b_strides, out, out_offset, out_shape, out_strides,
                 std::plus<float>());
}

void CpuBackend::sub(const DeviceBuffer &a, size_t a_offset,
                     const Shape &a_shape, const Strides &a_strides,
                     const DeviceBuffer &b, size_t b_offset,
                     const Shape &b_shape, const Strides &b_strides,
                     DeviceBuffer &out, size_t out_offset,
                     const Shape &out_shape, const Strides &out_strides) {
  elementwise_op(a, a_offset, a_shape, a_strides, b, b_offset, b_shape,
                 b_strides, out, out_offset, out_shape, out_strides,
                 std::minus<float>());
}

void CpuBackend::mul(const DeviceBuffer &a, size_t a_offset,
                     const Shape &a_shape, const Strides &a_strides,
                     const DeviceBuffer &b, size_t b_offset,
                     const Shape &b_shape, const Strides &b_strides,
                     DeviceBuffer &out, size_t out_offset,
                     const Shape &out_shape, const Strides &out_strides) {
  elementwise_op(a, a_offset, a_shape, a_strides, b, b_offset, b_shape,
                 b_strides, out, out_offset, out_shape, out_strides,
                 std::multiplies<float>());
}

void CpuBackend::div(const DeviceBuffer &a, size_t a_offset,
                     const Shape &a_shape, const Strides &a_strides,
                     const DeviceBuffer &b, size_t b_offset,
                     const Shape &b_shape, const Strides &b_strides,
                     DeviceBuffer &out, size_t out_offset,
                     const Shape &out_shape, const Strides &out_strides) {
  elementwise_op(a, a_offset, a_shape, a_strides, b, b_offset, b_shape,
                 b_strides, out, out_offset, out_shape, out_strides,
                 std::divides<float>());
}

void CpuBackend::matmul(const DeviceBuffer &a, size_t a_offset,
                        const Shape &a_shape, const Strides &a_strides,
                        const DeviceBuffer &b, size_t b_offset,
                        const Shape &b_shape, const Strides &b_strides,
                        DeviceBuffer &out, size_t out_offset,
                        const Shape &out_shape, const Strides &out_strides) {
  if (a_shape.size() != 2 || b_shape.size() != 2 || out_shape.size() != 2) {
    throw std::runtime_error("Matmul only supports 2D matrices.");
  }

  size_t M = a_shape[0];
  size_t K = a_shape[1];
  size_t N = b_shape[1];

  if (b_shape[0] != K || out_shape[0] != M || out_shape[1] != N) {
    throw std::runtime_error("Incompatible shapes for matmul.");
  }

  DataType a_dt = a.dtype();
  DataType b_dt = b.dtype();
  DataType out_dt = out.dtype();

  const void *a_data = a.data();
  const void *b_data = b.data();
  void *out_data = out.data();

  if (a_dt == DataType::FP32 && b_dt == DataType::FP32 && out_dt == DataType::FP32) {
    bool is_a_rm = (a_strides[0] == K && a_strides[1] == 1);
    bool is_a_trans = (a_strides[0] == 1 && a_strides[1] == M);
    bool is_b_rm = (b_strides[0] == N && b_strides[1] == 1);
    bool is_b_trans = (b_strides[0] == 1 && b_strides[1] == K);
    bool is_out_rm = (out_strides[0] == N && out_strides[1] == 1);

    if ((is_a_rm || is_a_trans) && (is_b_rm || is_b_trans) && is_out_rm) {
      int lda = is_a_rm ? static_cast<int>(K) : static_cast<int>(M);
      int ldb = is_b_rm ? static_cast<int>(N) : static_cast<int>(K);
      int ldc = static_cast<int>(N);
      int transA = is_a_rm ? CblasNoTrans : CblasTrans;
      int transB = is_b_rm ? CblasNoTrans : CblasTrans;

      cblas_sgemm(CblasRowMajor, transA, transB, static_cast<int>(M),
                  static_cast<int>(N), static_cast<int>(K), 1.0f,
                  static_cast<const float *>(a_data) + a_offset, lda,
                  static_cast<const float *>(b_data) + b_offset, ldb, 0.0f,
                  static_cast<float *>(out_data) + out_offset, ldc);
      return;
    }
  }

  for (size_t r = 0; r < M; ++r) {
    for (size_t c = 0; c < N; ++c) {
      float sum_val = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        size_t a_idx = a_offset + r * a_strides[0] + k * a_strides[1];
        size_t b_idx = b_offset + k * b_strides[0] + c * b_strides[1];
        sum_val += read_value(a_data, a_idx, a_dt) * read_value(b_data, b_idx, b_dt);
      }
      size_t out_idx = out_offset + r * out_strides[0] + c * out_strides[1];
      write_value(out_data, out_idx, sum_val, out_dt);
    }
  }
}

void CpuBackend::sum(const DeviceBuffer &input, size_t input_offset,
                     const Shape &input_shape, const Strides &input_strides,
                     DeviceBuffer &output, size_t output_offset,
                     const Shape &output_shape, const Strides &output_strides,
                     const std::vector<size_t> &axes) {
  size_t out_elements = count_elements(output_shape);
  void *out_data = output.data();
  DataType out_dt = output.dtype();

  for (size_t i = 0; i < out_elements; ++i) {
    size_t out_idx =
        get_input_offset(i, output_shape, output_shape, output_strides);
    write_value(out_data, output_offset + out_idx, 0.0f, out_dt);
  }

  size_t in_elements = count_elements(input_shape);
  const void *in_data = input.data();
  DataType in_dt = input.dtype();

  for (size_t i = 0; i < in_elements; ++i) {
    size_t temp = i;
    Shape in_coords(input_shape.size());
    for (size_t d = input_shape.size(); d > 0; --d) {
      in_coords[d - 1] = temp % input_shape[d - 1];
      temp /= input_shape[d - 1];
    }

    Shape out_coords;
    if (output_shape.size() == input_shape.size()) {
      out_coords.resize(input_shape.size());
      for (size_t d = 0; d < input_shape.size(); ++d) {
        bool is_summed = false;
        for (size_t axis : axes) {
          if (axis == d) {
            is_summed = true;
            break;
          }
        }
        out_coords[d] = is_summed ? 0 : in_coords[d];
      }
    } else {
      out_coords.reserve(output_shape.size());
      for (size_t d = 0; d < input_shape.size(); ++d) {
        bool is_summed = false;
        for (size_t axis : axes) {
          if (axis == d) {
            is_summed = true;
            break;
          }
        }
        if (!is_summed) {
          out_coords.push_back(in_coords[d]);
        }
      }
    }

    size_t out_idx = 0;
    for (size_t d = 0; d < out_coords.size(); ++d) {
      out_idx += out_coords[d] * output_strides[d];
    }

    size_t in_idx = 0;
    for (size_t d = 0; d < input_shape.size(); ++d) {
      in_idx += in_coords[d] * input_strides[d];
    }

    float current_val = read_value(out_data, output_offset + out_idx, out_dt);
    float input_val = read_value(in_data, input_offset + in_idx, in_dt);
    write_value(out_data, output_offset + out_idx, current_val + input_val, out_dt);
  }
}

void CpuBackend::relu(const DeviceBuffer &in, size_t in_offset,
                      const Shape &in_shape, const Strides &in_strides,
                      DeviceBuffer &out, size_t out_offset,
                      const Shape &out_shape, const Strides &out_strides) {
  unary_op(in, in_offset, in_shape, in_strides, out, out_offset, out_shape, out_strides,
           [](float x) { return std::max(x, 0.0f); });
}

void CpuBackend::relu_backward(const DeviceBuffer &in, size_t in_offset,
                               const Shape &in_shape, const Strides &in_strides,
                               const DeviceBuffer &grad_out,
                               size_t grad_out_offset,
                               const Shape &grad_out_shape,
                               const Strides &grad_out_strides,
                               DeviceBuffer &grad_in, size_t grad_in_offset,
                               const Shape &grad_in_shape,
                               const Strides &grad_in_strides) {
  binary_backward_op(in, in_offset, in_shape, in_strides, grad_out, grad_out_offset, grad_out_shape, grad_out_strides,
                     grad_in, grad_in_offset, grad_in_shape, grad_in_strides,
                     [](float x, float go) { return x > 0.0f ? go : 0.0f; });
}

void CpuBackend::sigmoid(const DeviceBuffer &in, size_t in_offset,
                         const Shape &in_shape, const Strides &in_strides,
                         DeviceBuffer &out, size_t out_offset,
                         const Shape &out_shape, const Strides &out_strides) {
  unary_op(in, in_offset, in_shape, in_strides, out, out_offset, out_shape, out_strides,
           [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
}

void CpuBackend::sigmoid_backward(
    const DeviceBuffer &out, size_t out_offset, const Shape &out_shape,
    const Strides &out_strides, const DeviceBuffer &grad_out,
    size_t grad_out_offset, const Shape &grad_out_shape,
    const Strides &grad_out_strides, DeviceBuffer &grad_in,
    size_t grad_in_offset, const Shape &grad_in_shape,
    const Strides &grad_in_strides) {
  binary_backward_op(out, out_offset, out_shape, out_strides, grad_out, grad_out_offset, grad_out_shape, grad_out_strides,
                     grad_in, grad_in_offset, grad_in_shape, grad_in_strides,
                     [](float y, float go) { return go * y * (1.0f - y); });
}

void CpuBackend::tanh(const DeviceBuffer &in, size_t in_offset,
                      const Shape &in_shape, const Strides &in_strides,
                      DeviceBuffer &out, size_t out_offset,
                      const Shape &out_shape, const Strides &out_strides) {
  unary_op(in, in_offset, in_shape, in_strides, out, out_offset, out_shape, out_strides,
           [](float x) { return std::tanh(x); });
}

void CpuBackend::tanh_backward(
    const DeviceBuffer &out, size_t out_offset, const Shape &out_shape,
    const Strides &out_strides, const DeviceBuffer &grad_out,
    size_t grad_out_offset, const Shape &grad_out_shape,
    const Strides &grad_out_strides, DeviceBuffer &grad_in,
    size_t grad_in_offset, const Shape &grad_in_shape,
    const Strides &grad_in_strides) {
  binary_backward_op(out, out_offset, out_shape, out_strides, grad_out, grad_out_offset, grad_out_shape, grad_out_strides,
                     grad_in, grad_in_offset, grad_in_shape, grad_in_strides,
                     [](float y, float go) { return go * (1.0f - y * y); });
}

void CpuBackend::softmax(const DeviceBuffer &in, size_t in_offset,
                         const Shape &in_shape, const Strides &in_strides,
                         DeviceBuffer &out, size_t out_offset,
                         const Shape &out_shape, const Strides &out_strides,
                         size_t axis) {
  const void *in_data = in.data();
  void *out_data = out.data();
  DataType in_dt = in.dtype();
  DataType out_dt = out.dtype();

  size_t num_elements = count_elements(in_shape);
  size_t axis_size = in_shape[axis];

  std::vector<bool> visited(num_elements, false);
  for (size_t i = 0; i < num_elements; ++i) {
    if (visited[i])
      continue;

    size_t temp = i;
    Shape coords(in_shape.size());
    for (size_t d = in_shape.size(); d > 0; --d) {
      coords[d - 1] = temp % in_shape[d - 1];
      temp /= in_shape[d - 1];
    }

    std::vector<size_t> slice_indices;
    slice_indices.reserve(axis_size);
    for (size_t k = 0; k < axis_size; ++k) {
      coords[axis] = k;
      size_t flat_idx = 0;
      for (size_t d = 0; d < in_shape.size(); ++d) {
        flat_idx = flat_idx * in_shape[d] + coords[d];
      }
      slice_indices.push_back(flat_idx);
      visited[flat_idx] = true;
    }

    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t idx : slice_indices) {
      size_t in_offset_idx =
          get_input_offset(idx, in_shape, in_shape, in_strides);
      max_val = std::max(max_val, read_value(in_data, in_offset + in_offset_idx, in_dt));
    }

    float sum_exp = 0.0f;
    for (size_t idx : slice_indices) {
      size_t in_offset_idx =
          get_input_offset(idx, in_shape, in_shape, in_strides);
      sum_exp += std::exp(read_value(in_data, in_offset + in_offset_idx, in_dt) - max_val);
    }

    for (size_t idx : slice_indices) {
      size_t in_offset_idx =
          get_input_offset(idx, in_shape, in_shape, in_strides);
      size_t out_offset_idx =
          get_input_offset(idx, out_shape, out_shape, out_strides);
      float out_val = std::exp(read_value(in_data, in_offset + in_offset_idx, in_dt) - max_val) / sum_exp;
      write_value(out_data, out_offset + out_offset_idx, out_val, out_dt);
    }
  }
}

void CpuBackend::softmax_backward(
    const DeviceBuffer &out, size_t out_offset, const Shape &out_shape,
    const Strides &out_strides, const DeviceBuffer &grad_out,
    size_t grad_out_offset, const Shape &grad_out_shape,
    const Strides &grad_out_strides, DeviceBuffer &grad_in,
    size_t grad_in_offset, const Shape &grad_in_shape,
    const Strides &grad_in_strides, size_t axis) {
  const void *out_data = out.data();
  const void *go_data = grad_out.data();
  void *gi_data = grad_in.data();
  DataType out_dt = out.dtype();
  DataType go_dt = grad_out.dtype();
  DataType gi_dt = grad_in.dtype();

  size_t num_elements = count_elements(out_shape);
  size_t axis_size = out_shape[axis];

  std::vector<bool> visited(num_elements, false);
  for (size_t i = 0; i < num_elements; ++i) {
    if (visited[i])
      continue;

    size_t temp = i;
    Shape coords(out_shape.size());
    for (size_t d = out_shape.size(); d > 0; --d) {
      coords[d - 1] = temp % out_shape[d - 1];
      temp /= out_shape[d - 1];
    }

    std::vector<size_t> slice_indices;
    slice_indices.reserve(axis_size);
    for (size_t k = 0; k < axis_size; ++k) {
      coords[axis] = k;
      size_t flat_idx = 0;
      for (size_t d = 0; d < out_shape.size(); ++d) {
        flat_idx = flat_idx * out_shape[d] + coords[d];
      }
      slice_indices.push_back(flat_idx);
      visited[flat_idx] = true;
    }

    float sum_dy_y = 0.0f;
    for (size_t idx : slice_indices) {
      size_t out_offset_idx =
          get_input_offset(idx, out_shape, out_shape, out_strides);
      size_t go_offset_idx =
          get_input_offset(idx, out_shape, grad_out_shape, grad_out_strides);
      sum_dy_y += read_value(go_data, grad_out_offset + go_offset_idx, go_dt) *
                  read_value(out_data, out_offset + out_offset_idx, out_dt);
    }

    for (size_t idx : slice_indices) {
      size_t out_offset_idx =
          get_input_offset(idx, out_shape, out_shape, out_strides);
      size_t go_offset_idx =
          get_input_offset(idx, out_shape, grad_out_shape, grad_out_strides);
      size_t gi_offset_idx =
          get_input_offset(idx, out_shape, grad_in_shape, grad_in_strides);
      float y = read_value(out_data, out_offset + out_offset_idx, out_dt);
      float dy = read_value(go_data, grad_out_offset + go_offset_idx, go_dt);
      write_value(gi_data, grad_in_offset + gi_offset_idx, y * (dy - sum_dy_y), gi_dt);
    }
  }
}

void CpuBackend::sqrt(const DeviceBuffer &in, size_t in_offset,
                      const Shape &in_shape, const Strides &in_strides,
                      DeviceBuffer &out, size_t out_offset,
                      const Shape &out_shape, const Strides &out_strides) {
  unary_op(in, in_offset, in_shape, in_strides, out, out_offset, out_shape, out_strides,
           [](float x) { return std::sqrt(x); });
}

void CpuBackend::sqrt_backward(
    const DeviceBuffer &out, size_t out_offset, const Shape &out_shape,
    const Strides &out_strides, const DeviceBuffer &grad_out,
    size_t grad_out_offset, const Shape &grad_out_shape,
    const Strides &grad_out_strides, DeviceBuffer &grad_in,
    size_t grad_in_offset, const Shape &grad_in_shape,
    const Strides &grad_in_strides) {
  binary_backward_op(out, out_offset, out_shape, out_strides, grad_out, grad_out_offset, grad_out_shape, grad_out_strides,
                     grad_in, grad_in_offset, grad_in_shape, grad_in_strides,
                     [](float y, float go) { return go * (0.5f / (y + 1e-8f)); });
}

void CpuBackend::log(const DeviceBuffer &in, size_t in_offset,
                     const Shape &in_shape, const Strides &in_strides,
                     DeviceBuffer &out, size_t out_offset,
                     const Shape &out_shape, const Strides &out_strides) {
  unary_op(in, in_offset, in_shape, in_strides, out, out_offset, out_shape, out_strides,
           [](float x) { return std::log(x + 1e-8f); });
}

void CpuBackend::log_backward(const DeviceBuffer &in, size_t in_offset,
                              const Shape &in_shape, const Strides &in_strides,
                              const DeviceBuffer &grad_out,
                              size_t grad_out_offset,
                              const Shape &grad_out_shape,
                              const Strides &grad_out_strides,
                              DeviceBuffer &grad_in, size_t grad_in_offset,
                              const Shape &grad_in_shape,
                              const Strides &grad_in_strides) {
  binary_backward_op(in, in_offset, in_shape, in_strides, grad_out, grad_out_offset, grad_out_shape, grad_out_strides,
                     grad_in, grad_in_offset, grad_in_shape, grad_in_strides,
                     [](float x, float go) { return go / (x + 1e-8f); });
}

void CpuBackend::conv2d(const DeviceBuffer &in, size_t in_offset,
                        const Shape &in_shape, const Strides &in_strides,
                        const DeviceBuffer &weight, size_t weight_offset,
                        const Shape &weight_shape,
                        const Strides &weight_strides, const DeviceBuffer &bias,
                        size_t bias_offset, const Shape &bias_shape,
                        const Strides &bias_strides, DeviceBuffer &out,
                        size_t out_offset, const Shape &out_shape,
                        const Strides &out_strides, size_t padding,
                        size_t stride) {
  const float *in_data = static_cast<const float *>(in.data()) + in_offset;
  const float *w_data =
      static_cast<const float *>(weight.data()) + weight_offset;
  const float *b_data = static_cast<const float *>(bias.data()) + bias_offset;
  float *out_data = static_cast<float *>(out.data()) + out_offset;

  size_t N = in_shape[0];
  size_t C_in = in_shape[1];
  size_t H = in_shape[2];
  size_t W = in_shape[3];

  size_t C_out = weight_shape[0];
  size_t K_h = weight_shape[2];
  size_t K_w = weight_shape[3];

  size_t H_out = out_shape[2];
  size_t W_out = out_shape[3];

  size_t sN_in = in_strides[0], sC_in = in_strides[1], sH_in = in_strides[2],
         sW_in = in_strides[3];
  size_t sCo_w = weight_strides[0], sCi_w = weight_strides[1],
         sKh_w = weight_strides[2], sKw_w = weight_strides[3];
  size_t sCo_b = (bias_shape.size() > 0) ? bias_strides[0] : 0;
  size_t sN_out = out_strides[0], sCo_out = out_strides[1],
         sH_out = out_strides[2], sW_out = out_strides[3];

  for (size_t n = 0; n < N; ++n) {
    for (size_t co = 0; co < C_out; ++co) {
      float b = b_data[co * sCo_b];
      for (size_t ho = 0; ho < H_out; ++ho) {
        for (size_t wo = 0; wo < W_out; ++wo) {
          float sum = 0.0f;
          for (size_t ci = 0; ci < C_in; ++ci) {
            for (size_t kh = 0; kh < K_h; ++kh) {
              int h_in = (int)(ho * stride) - (int)padding + (int)kh;
              if (h_in < 0 || h_in >= (int)H)
                continue;
              for (size_t kw = 0; kw < K_w; ++kw) {
                int w_in = (int)(wo * stride) - (int)padding + (int)kw;
                if (w_in < 0 || w_in >= (int)W)
                  continue;

                size_t in_idx = n * sN_in + ci * sC_in + (size_t)h_in * sH_in +
                                (size_t)w_in * sW_in;
                size_t w_idx =
                    co * sCo_w + ci * sCi_w + kh * sKh_w + kw * sKw_w;
                sum += in_data[in_idx] * w_data[w_idx];
              }
            }
          }
          size_t out_idx =
              n * sN_out + co * sCo_out + ho * sH_out + wo * sW_out;
          out_data[out_idx] = sum + b;
        }
      }
    }
  }
}

void CpuBackend::conv2d_backward(
    const DeviceBuffer &in, size_t in_offset, const Shape &in_shape,
    const Strides &in_strides, const DeviceBuffer &weight, size_t weight_offset,
    const Shape &weight_shape, const Strides &weight_strides,
    const DeviceBuffer &grad_out, size_t grad_out_offset,
    const Shape &grad_out_shape, const Strides &grad_out_strides,
    DeviceBuffer &grad_in, size_t grad_in_offset, const Shape &grad_in_shape,
    const Strides &grad_in_strides, DeviceBuffer &grad_weight,
    size_t grad_weight_offset, const Shape &grad_weight_shape,
    const Strides &grad_weight_strides, DeviceBuffer &grad_bias,
    size_t grad_bias_offset, const Shape &grad_bias_shape,
    const Strides &grad_bias_strides, size_t padding, size_t stride) {
  size_t total_gi = count_elements(grad_in_shape);
  float *gi_data = static_cast<float *>(grad_in.data()) + grad_in_offset;
  for (size_t i = 0; i < total_gi; ++i) {
    size_t idx =
        get_input_offset(i, grad_in_shape, grad_in_shape, grad_in_strides);
    gi_data[idx] = 0.0f;
  }

  size_t total_gw = count_elements(grad_weight_shape);
  float *gw_data =
      static_cast<float *>(grad_weight.data()) + grad_weight_offset;
  for (size_t i = 0; i < total_gw; ++i) {
    size_t idx = get_input_offset(i, grad_weight_shape, grad_weight_shape,
                                  grad_weight_strides);
    gw_data[idx] = 0.0f;
  }

  size_t total_gb = count_elements(grad_bias_shape);
  float *gb_data = static_cast<float *>(grad_bias.data()) + grad_bias_offset;
  for (size_t i = 0; i < total_gb; ++i) {
    size_t idx = get_input_offset(i, grad_bias_shape, grad_bias_shape,
                                  grad_bias_strides);
    gb_data[idx] = 0.0f;
  }

  size_t N = in_shape[0];
  size_t C_in = in_shape[1];
  size_t H = in_shape[2];
  size_t W = in_shape[3];

  size_t C_out = weight_shape[0];
  size_t K_h = weight_shape[2];
  size_t K_w = weight_shape[3];

  size_t H_out = grad_out_shape[2];
  size_t W_out = grad_out_shape[3];

  size_t sN_in = in_strides[0], sC_in = in_strides[1], sH_in = in_strides[2],
         sW_in = in_strides[3];
  size_t sCo_w = weight_strides[0], sCi_w = weight_strides[1],
         sKh_w = weight_strides[2], sKw_w = weight_strides[3];

  size_t sN_go = grad_out_strides[0], sCo_go = grad_out_strides[1],
         sH_go = grad_out_strides[2], sW_go = grad_out_strides[3];
  size_t sN_gi = grad_in_strides[0], sC_gi = grad_in_strides[1],
         sH_gi = grad_in_strides[2], sW_gi = grad_in_strides[3];
  size_t sCo_gw = grad_weight_strides[0], sCi_gw = grad_weight_strides[1],
         sKh_gw = grad_weight_strides[2], sKw_gw = grad_weight_strides[3];
  size_t sCo_gb = (grad_bias_shape.size() > 0) ? grad_bias_strides[0] : 0;

  const float *go_data =
      static_cast<const float *>(grad_out.data()) + grad_out_offset;
  const float *in_data = static_cast<const float *>(in.data()) + in_offset;
  const float *w_data =
      static_cast<const float *>(weight.data()) + weight_offset;

  for (size_t n = 0; n < N; ++n) {
    for (size_t co = 0; co < C_out; ++co) {
      for (size_t ho = 0; ho < H_out; ++ho) {
        for (size_t wo = 0; wo < W_out; ++wo) {
          size_t go_idx = n * sN_go + co * sCo_go + ho * sH_go + wo * sW_go;
          float dy = go_data[go_idx];

          gb_data[co * sCo_gb] += dy;

          for (size_t ci = 0; ci < C_in; ++ci) {
            for (size_t kh = 0; kh < K_h; ++kh) {
              int h_in = (int)(ho * stride) - (int)padding + (int)kh;
              if (h_in < 0 || h_in >= (int)H)
                continue;
              for (size_t kw = 0; kw < K_w; ++kw) {
                int w_in = (int)(wo * stride) - (int)padding + (int)kw;
                if (w_in < 0 || w_in >= (int)W)
                  continue;

                size_t in_idx = n * sN_in + ci * sC_in + (size_t)h_in * sH_in +
                                (size_t)w_in * sW_in;
                size_t w_idx =
                    co * sCo_w + ci * sCi_w + kh * sKh_w + kw * sKw_w;

                size_t gi_idx = n * sN_gi + ci * sC_gi + (size_t)h_in * sH_gi +
                                (size_t)w_in * sW_gi;
                size_t gw_idx =
                    co * sCo_gw + ci * sCi_gw + kh * sKh_gw + kw * sKw_gw;

                gi_data[gi_idx] += dy * w_data[w_idx];
                gw_data[gw_idx] += dy * in_data[in_idx];
              }
            }
          }
        }
      }
    }
  }
}

void CpuBackend::maxpool2d(const DeviceBuffer &in, size_t in_offset,
                           const Shape &in_shape, const Strides &in_strides,
                           DeviceBuffer &out, size_t out_offset,
                           const Shape &out_shape, const Strides &out_strides,
                           size_t pool_h, size_t pool_w, size_t stride) {
  const float *in_data = static_cast<const float *>(in.data()) + in_offset;
  float *out_data = static_cast<float *>(out.data()) + out_offset;

  size_t N = in_shape[0];
  size_t C = in_shape[1];
  size_t H = in_shape[2];
  size_t W = in_shape[3];

  size_t H_out = out_shape[2];
  size_t W_out = out_shape[3];

  size_t sN_in = in_strides[0], sC_in = in_strides[1], sH_in = in_strides[2],
         sW_in = in_strides[3];
  size_t sN_out = out_strides[0], sC_out = out_strides[1],
         sH_out = out_strides[2], sW_out = out_strides[3];

  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
      for (size_t ho = 0; ho < H_out; ++ho) {
        for (size_t wo = 0; wo < W_out; ++wo) {
          float max_val = -std::numeric_limits<float>::infinity();
          for (size_t kh = 0; kh < pool_h; ++kh) {
            size_t h_in = ho * stride + kh;
            if (h_in >= H)
              continue;
            for (size_t kw = 0; kw < pool_w; ++kw) {
              size_t w_in = wo * stride + kw;
              if (w_in >= W)
                continue;

              size_t in_idx =
                  n * sN_in + c * sC_in + h_in * sH_in + w_in * sW_in;
              if (in_data[in_idx] > max_val) {
                max_val = in_data[in_idx];
              }
            }
          }
          size_t out_idx = n * sN_out + c * sC_out + ho * sH_out + wo * sW_out;
          out_data[out_idx] = max_val;
        }
      }
    }
  }
}

void CpuBackend::maxpool2d_backward(
    const DeviceBuffer &in, size_t in_offset, const Shape &in_shape,
    const Strides &in_strides, const DeviceBuffer &out, size_t out_offset,
    const Shape &out_shape, const Strides &out_strides,
    const DeviceBuffer &grad_out, size_t grad_out_offset,
    [[maybe_unused]] const Shape &grad_out_shape,
    const Strides &grad_out_strides, DeviceBuffer &grad_in,
    size_t grad_in_offset, const Shape &grad_in_shape,
    const Strides &grad_in_strides, size_t pool_h, size_t pool_w,
    size_t stride) {
  size_t total_gi = count_elements(grad_in_shape);
  float *gi_data = static_cast<float *>(grad_in.data()) + grad_in_offset;
  for (size_t i = 0; i < total_gi; ++i) {
    size_t idx =
        get_input_offset(i, grad_in_shape, grad_in_shape, grad_in_strides);
    gi_data[idx] = 0.0f;
  }

  size_t N = in_shape[0];
  size_t C = in_shape[1];
  size_t H = in_shape[2];
  size_t W = in_shape[3];

  size_t H_out = out_shape[2];
  size_t W_out = out_shape[3];

  size_t sN_in = in_strides[0], sC_in = in_strides[1], sH_in = in_strides[2],
         sW_in = in_strides[3];
  size_t sN_out = out_strides[0], sC_out = out_strides[1],
         sH_out = out_strides[2], sW_out = out_strides[3];
  size_t sN_go = grad_out_strides[0], sC_go = grad_out_strides[1],
         sH_go = grad_out_strides[2], sW_go = grad_out_strides[3];
  size_t sN_gi = grad_in_strides[0], sC_gi = grad_in_strides[1],
         sH_gi = grad_in_strides[2], sW_gi = grad_in_strides[3];

  const float *in_data = static_cast<const float *>(in.data()) + in_offset;
  const float *out_data = static_cast<const float *>(out.data()) + out_offset;
  const float *go_data =
      static_cast<const float *>(grad_out.data()) + grad_out_offset;

  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
      for (size_t ho = 0; ho < H_out; ++ho) {
        for (size_t wo = 0; wo < W_out; ++wo) {
          size_t out_idx = n * sN_out + c * sC_out + ho * sH_out + wo * sW_out;
          float max_val = out_data[out_idx];

          size_t go_idx = n * sN_go + c * sC_go + ho * sH_go + wo * sW_go;
          float dy = go_data[go_idx];

          bool found = false;
          for (size_t kh = 0; kh < pool_h; ++kh) {
            size_t h_in = ho * stride + kh;
            if (h_in >= H)
              continue;
            for (size_t kw = 0; kw < pool_w; ++kw) {
              size_t w_in = wo * stride + kw;
              if (w_in >= W)
                continue;

              size_t in_idx =
                  n * sN_in + c * sC_in + h_in * sH_in + w_in * sW_in;
              if (std::abs(in_data[in_idx] - max_val) < 1e-6f) {
                size_t gi_idx =
                    n * sN_gi + c * sC_gi + h_in * sH_gi + w_in * sW_gi;
                gi_data[gi_idx] += dy;
                found = true;
                break;
              }
            }
            if (found)
              break;
          }
        }
      }
    }
  }
}

void CpuBackend::embedding(
    const DeviceBuffer &weight, size_t weight_offset, const Shape &weight_shape,
    const Strides &weight_strides, const DeviceBuffer &indices,
    size_t indices_offset, const Shape &indices_shape,
    const Strides &indices_strides, DeviceBuffer &out, size_t out_offset,
    [[maybe_unused]] const Shape &out_shape, const Strides &out_strides) {
  const float *w_data =
      static_cast<const float *>(weight.data()) + weight_offset;
  const float *idx_data =
      static_cast<const float *>(indices.data()) + indices_offset;
  float *out_data = static_cast<float *>(out.data()) + out_offset;

  size_t N = indices_shape[0];
  size_t T = indices_shape[1];
  size_t D = weight_shape[1];

  size_t sN_idx = indices_strides[0], sT_idx = indices_strides[1];
  size_t sV_w = weight_strides[0], sD_w = weight_strides[1];
  size_t sN_out = out_strides[0], sT_out = out_strides[1],
         sD_out = out_strides[2];

  for (size_t n = 0; n < N; ++n) {
    for (size_t t = 0; t < T; ++t) {
      size_t idx_offset = n * sN_idx + t * sT_idx;
      size_t vocab_idx = (size_t)idx_data[idx_offset];
      if (vocab_idx >= weight_shape[0]) {
        vocab_idx = 0;
      }
      for (size_t d = 0; d < D; ++d) {
        size_t out_idx = n * sN_out + t * sT_out + d * sD_out;
        size_t w_idx = vocab_idx * sV_w + d * sD_w;
        out_data[out_idx] = w_data[w_idx];
      }
    }
  }
}

void CpuBackend::embedding_backward(
    const DeviceBuffer &grad_out, size_t grad_out_offset,
    [[maybe_unused]] const Shape &grad_out_shape,
    const Strides &grad_out_strides, const DeviceBuffer &indices,
    size_t indices_offset, const Shape &indices_shape,
    const Strides &indices_strides, DeviceBuffer &grad_weight,
    size_t grad_weight_offset, const Shape &grad_weight_shape,
    const Strides &grad_weight_strides) {
  size_t total_gw = count_elements(grad_weight_shape);
  float *gw_data =
      static_cast<float *>(grad_weight.data()) + grad_weight_offset;
  for (size_t i = 0; i < total_gw; ++i) {
    size_t idx = get_input_offset(i, grad_weight_shape, grad_weight_shape,
                                  grad_weight_strides);
    gw_data[idx] = 0.0f;
  }

  size_t N = indices_shape[0];
  size_t T = indices_shape[1];
  size_t D = grad_weight_shape[1];

  size_t sN_idx = indices_strides[0], sT_idx = indices_strides[1];
  size_t sV_gw = grad_weight_strides[0], sD_gw = grad_weight_strides[1];
  size_t sN_go = grad_out_strides[0], sT_go = grad_out_strides[1],
         sD_go = grad_out_strides[2];

  const float *go_data =
      static_cast<const float *>(grad_out.data()) + grad_out_offset;
  const float *idx_data =
      static_cast<const float *>(indices.data()) + indices_offset;

  for (size_t n = 0; n < N; ++n) {
    for (size_t t = 0; t < T; ++t) {
      size_t idx_offset = n * sN_idx + t * sT_idx;
      size_t vocab_idx = (size_t)idx_data[idx_offset];
      if (vocab_idx >= grad_weight_shape[0]) {
        continue;
      }
      for (size_t d = 0; d < D; ++d) {
        size_t go_idx = n * sN_go + t * sT_go + d * sD_go;
        size_t gw_idx = vocab_idx * sV_gw + d * sD_gw;
        gw_data[gw_idx] += go_data[go_idx];
      }
    }
  }
}

void CpuBackend::slice_backward(const DeviceBuffer &grad_slice,
                                size_t gs_offset, const Shape &gs_shape,
                                const Strides &gs_strides,
                                DeviceBuffer &grad_parent, size_t gp_offset,
                                const Shape &gp_shape,
                                const Strides &gp_strides, size_t axis,
                                size_t index) {
  const float *gs_data =
      static_cast<const float *>(grad_slice.data()) + gs_offset;
  float *gp_data = static_cast<float *>(grad_parent.data()) + gp_offset;

  size_t total = count_elements(gs_shape);
  for (size_t i = 0; i < total; ++i) {
    size_t temp = i;
    Shape slice_coords(gs_shape.size());
    for (size_t d = gs_shape.size(); d > 0; --d) {
      slice_coords[d - 1] = temp % gs_shape[d - 1];
      temp /= gs_shape[d - 1];
    }

    Shape parent_coords(gp_shape.size());
    size_t slice_d = 0;
    for (size_t d = 0; d < gp_shape.size(); ++d) {
      if (d == axis) {
        parent_coords[d] = index;
      } else {
        parent_coords[d] = slice_coords[slice_d++];
      }
    }

    size_t slice_idx = get_input_offset(i, gs_shape, gs_shape, gs_strides);
    size_t parent_idx = 0;
    for (size_t d = 0; d < gp_shape.size(); ++d) {
      parent_idx += parent_coords[d] * gp_strides[d];
    }

    gp_data[parent_idx] += gs_data[slice_idx];
  }
}
