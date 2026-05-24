#ifndef BACKEND_HPP
#define BACKEND_HPP

#include "core/shape.hpp"
#include <memory>
#include <vector>

class DeviceBuffer {
public:
  virtual ~DeviceBuffer() = default;
  [[nodiscard]] virtual size_t size() const noexcept = 0;
  [[nodiscard]] virtual void *data() noexcept = 0;
  [[nodiscard]] virtual const void *data() const noexcept = 0;
};

class ExecutionBackend {
public:
  virtual ~ExecutionBackend() = default;

  [[nodiscard]] virtual std::unique_ptr<DeviceBuffer>
  allocate(size_t bytes) = 0;

  virtual void copy_host_to_device(const float *host_ptr,
                                   DeviceBuffer &device_buf,
                                   size_t size_bytes) = 0;
  virtual void copy_to_device(const float *host_ptr, DeviceBuffer &device_buf,
                              size_t dest_offset_bytes, size_t size_bytes) = 0;
  virtual void copy_device_to_host(const DeviceBuffer &device_buf,
                                   float *host_ptr, size_t size_bytes) = 0;
  virtual void fill(DeviceBuffer &buffer, float value) = 0;

  virtual void add(const DeviceBuffer &a, size_t a_offset, const Shape &a_shape,
                   const Strides &a_strides, const DeviceBuffer &b,
                   size_t b_offset, const Shape &b_shape,
                   const Strides &b_strides, DeviceBuffer &out,
                   size_t out_offset, const Shape &out_shape,
                   const Strides &out_strides) = 0;

  virtual void sub(const DeviceBuffer &a, size_t a_offset, const Shape &a_shape,
                   const Strides &a_strides, const DeviceBuffer &b,
                   size_t b_offset, const Shape &b_shape,
                   const Strides &b_strides, DeviceBuffer &out,
                   size_t out_offset, const Shape &out_shape,
                   const Strides &out_strides) = 0;

  virtual void mul(const DeviceBuffer &a, size_t a_offset, const Shape &a_shape,
                   const Strides &a_strides, const DeviceBuffer &b,
                   size_t b_offset, const Shape &b_shape,
                   const Strides &b_strides, DeviceBuffer &out,
                   size_t out_offset, const Shape &out_shape,
                   const Strides &out_strides) = 0;

  virtual void div(const DeviceBuffer &a, size_t a_offset, const Shape &a_shape,
                   const Strides &a_strides, const DeviceBuffer &b,
                   size_t b_offset, const Shape &b_shape,
                   const Strides &b_strides, DeviceBuffer &out,
                   size_t out_offset, const Shape &out_shape,
                   const Strides &out_strides) = 0;

  virtual void matmul(const DeviceBuffer &a, size_t a_offset,
                      const Shape &a_shape, const Strides &a_strides,
                      const DeviceBuffer &b, size_t b_offset,
                      const Shape &b_shape, const Strides &b_strides,
                      DeviceBuffer &out, size_t out_offset,
                      const Shape &out_shape, const Strides &out_strides) = 0;

  virtual void sum(const DeviceBuffer &input, size_t input_offset,
                   const Shape &input_shape, const Strides &input_strides,
                   DeviceBuffer &output, size_t output_offset,
                   const Shape &output_shape, const Strides &output_strides,
                   const std::vector<size_t> &axes) = 0;

  virtual void relu(const DeviceBuffer &in, size_t in_offset,
                    const Shape &in_shape, const Strides &in_strides,
                    DeviceBuffer &out, size_t out_offset,
                    const Shape &out_shape, const Strides &out_strides) = 0;

  virtual void
  relu_backward(const DeviceBuffer &in, size_t in_offset, const Shape &in_shape,
                const Strides &in_strides, const DeviceBuffer &grad_out,
                size_t grad_out_offset, const Shape &grad_out_shape,
                const Strides &grad_out_strides, DeviceBuffer &grad_in,
                size_t grad_in_offset, const Shape &grad_in_shape,
                const Strides &grad_in_strides) = 0;

  virtual void sigmoid(const DeviceBuffer &in, size_t in_offset,
                       const Shape &in_shape, const Strides &in_strides,
                       DeviceBuffer &out, size_t out_offset,
                       const Shape &out_shape, const Strides &out_strides) = 0;

  virtual void
  sigmoid_backward(const DeviceBuffer &out, size_t out_offset,
                   const Shape &out_shape, const Strides &out_strides,
                   const DeviceBuffer &grad_out, size_t grad_out_offset,
                   const Shape &grad_out_shape, const Strides &grad_out_strides,
                   DeviceBuffer &grad_in, size_t grad_in_offset,
                   const Shape &grad_in_shape,
                   const Strides &grad_in_strides) = 0;

  virtual void tanh(const DeviceBuffer &in, size_t in_offset,
                    const Shape &in_shape, const Strides &in_strides,
                    DeviceBuffer &out, size_t out_offset,
                    const Shape &out_shape, const Strides &out_strides) = 0;

  virtual void
  tanh_backward(const DeviceBuffer &out, size_t out_offset,
                const Shape &out_shape, const Strides &out_strides,
                const DeviceBuffer &grad_out, size_t grad_out_offset,
                const Shape &grad_out_shape, const Strides &grad_out_strides,
                DeviceBuffer &grad_in, size_t grad_in_offset,
                const Shape &grad_in_shape, const Strides &grad_in_strides) = 0;

  virtual void softmax(const DeviceBuffer &in, size_t in_offset,
                       const Shape &in_shape, const Strides &in_strides,
                       DeviceBuffer &out, size_t out_offset,
                       const Shape &out_shape, const Strides &out_strides,
                       size_t axis) = 0;

  virtual void
  softmax_backward(const DeviceBuffer &out, size_t out_offset,
                   const Shape &out_shape, const Strides &out_strides,
                   const DeviceBuffer &grad_out, size_t grad_out_offset,
                   const Shape &grad_out_shape, const Strides &grad_out_strides,
                   DeviceBuffer &grad_in, size_t grad_in_offset,
                   const Shape &grad_in_shape, const Strides &grad_in_strides,
                   size_t axis) = 0;

  virtual void sqrt(const DeviceBuffer &in, size_t in_offset,
                    const Shape &in_shape, const Strides &in_strides,
                    DeviceBuffer &out, size_t out_offset,
                    const Shape &out_shape, const Strides &out_strides) = 0;

  virtual void
  sqrt_backward(const DeviceBuffer &out, size_t out_offset,
                const Shape &out_shape, const Strides &out_strides,
                const DeviceBuffer &grad_out, size_t grad_out_offset,
                const Shape &grad_out_shape, const Strides &grad_out_strides,
                DeviceBuffer &grad_in, size_t grad_in_offset,
                const Shape &grad_in_shape, const Strides &grad_in_strides) = 0;

  virtual void log(const DeviceBuffer &in, size_t in_offset,
                   const Shape &in_shape, const Strides &in_strides,
                   DeviceBuffer &out, size_t out_offset, const Shape &out_shape,
                   const Strides &out_strides) = 0;

  virtual void log_backward(const DeviceBuffer &in, size_t in_offset,
                            const Shape &in_shape, const Strides &in_strides,
                            const DeviceBuffer &grad_out,
                            size_t grad_out_offset, const Shape &grad_out_shape,
                            const Strides &grad_out_strides,
                            DeviceBuffer &grad_in, size_t grad_in_offset,
                            const Shape &grad_in_shape,
                            const Strides &grad_in_strides) = 0;

  virtual void conv2d(const DeviceBuffer &in, size_t in_offset,
                      const Shape &in_shape, const Strides &in_strides,
                      const DeviceBuffer &weight, size_t weight_offset,
                      const Shape &weight_shape, const Strides &weight_strides,
                      const DeviceBuffer &bias, size_t bias_offset,
                      const Shape &bias_shape, const Strides &bias_strides,
                      DeviceBuffer &out, size_t out_offset,
                      const Shape &out_shape, const Strides &out_strides,
                      size_t padding, size_t stride) = 0;

  virtual void conv2d_backward(
      const DeviceBuffer &in, size_t in_offset, const Shape &in_shape,
      const Strides &in_strides, const DeviceBuffer &weight,
      size_t weight_offset, const Shape &weight_shape,
      const Strides &weight_strides, const DeviceBuffer &grad_out,
      size_t grad_out_offset, const Shape &grad_out_shape,
      const Strides &grad_out_strides, DeviceBuffer &grad_in,
      size_t grad_in_offset, const Shape &grad_in_shape,
      const Strides &grad_in_strides, DeviceBuffer &grad_weight,
      size_t grad_weight_offset, const Shape &grad_weight_shape,
      const Strides &grad_weight_strides, DeviceBuffer &grad_bias,
      size_t grad_bias_offset, const Shape &grad_bias_shape,
      const Strides &grad_bias_strides, size_t padding, size_t stride) = 0;

  virtual void maxpool2d(const DeviceBuffer &in, size_t in_offset,
                         const Shape &in_shape, const Strides &in_strides,
                         DeviceBuffer &out, size_t out_offset,
                         const Shape &out_shape, const Strides &out_strides,
                         size_t pool_h, size_t pool_w, size_t stride) = 0;

  virtual void maxpool2d_backward(
      const DeviceBuffer &in, size_t in_offset, const Shape &in_shape,
      const Strides &in_strides, const DeviceBuffer &out, size_t out_offset,
      const Shape &out_shape, const Strides &out_strides,
      const DeviceBuffer &grad_out, size_t grad_out_offset,
      const Shape &grad_out_shape, const Strides &grad_out_strides,
      DeviceBuffer &grad_in, size_t grad_in_offset, const Shape &grad_in_shape,
      const Strides &grad_in_strides, size_t pool_h, size_t pool_w,
      size_t stride) = 0;

  virtual void embedding(const DeviceBuffer &weight, size_t weight_offset,
                         const Shape &weight_shape,
                         const Strides &weight_strides,
                         const DeviceBuffer &indices, size_t indices_offset,
                         const Shape &indices_shape,
                         const Strides &indices_strides, DeviceBuffer &out,
                         size_t out_offset, const Shape &out_shape,
                         const Strides &out_strides) = 0;

  virtual void embedding_backward(
      const DeviceBuffer &grad_out, size_t grad_out_offset,
      const Shape &grad_out_shape, const Strides &grad_out_strides,
      const DeviceBuffer &indices, size_t indices_offset,
      const Shape &indices_shape, const Strides &indices_strides,
      DeviceBuffer &grad_weight, size_t grad_weight_offset,
      const Shape &grad_weight_shape, const Strides &grad_weight_strides) = 0;

  virtual void slice_backward(const DeviceBuffer &grad_slice, size_t gs_offset,
                              const Shape &gs_shape, const Strides &gs_strides,
                              DeviceBuffer &grad_parent, size_t gp_offset,
                              const Shape &gp_shape, const Strides &gp_strides,
                              size_t axis, size_t index) = 0;
};

#endif // BACKEND_HPP
