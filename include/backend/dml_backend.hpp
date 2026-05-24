#ifndef DML_BACKEND_HPP
#define DML_BACKEND_HPP

#include "backend/backend.hpp"

#ifdef WITH_DML
#include <d3d12.h>
#include <DirectML.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;
#endif

#include <memory>
#include <vector>
#include <stdexcept>

class DmlBuffer : public DeviceBuffer {
public:
  explicit DmlBuffer(size_t size_bytes, void *dml_device = nullptr);
  ~DmlBuffer() override;

  [[nodiscard]] size_t size() const noexcept override { return size_; }
  [[nodiscard]] void *data() noexcept override;
  [[nodiscard]] const void *data() const noexcept override;

  [[nodiscard]] DataType dtype() const noexcept override { return dtype_; }
  void set_dtype(DataType dt) noexcept override { dtype_ = dt; }

#ifdef WITH_DML
  ComPtr<ID3D12Resource> resource() const { return resource_; }
#endif

private:
  size_t size_;
  DataType dtype_ = DataType::FP32;
#ifdef WITH_DML
  ComPtr<ID3D12Resource> resource_;
#endif
};

class DmlBackend : public ExecutionBackend {
public:
  DmlBackend();
  ~DmlBackend() override;

  [[nodiscard]] std::unique_ptr<DeviceBuffer> allocate(size_t bytes) override;

  void copy_host_to_device(const float *host_ptr, DeviceBuffer &device_buf,
                           size_t size_bytes) override;
  void copy_to_device(const float *host_ptr, DeviceBuffer &device_buf,
                      size_t dest_offset_bytes, size_t size_bytes) override;
  void copy_device_to_host(const DeviceBuffer &device_buf, float *host_ptr,
                           size_t size_bytes) override;
  void fill(DeviceBuffer &buffer, float value) override;

  void add(const DeviceBuffer &a, size_t a_offset, const Shape &a_shape,
           const Strides &a_strides, const DeviceBuffer &b, size_t b_offset,
           const Shape &b_shape, const Strides &b_strides, DeviceBuffer &out,
           size_t out_offset, const Shape &out_shape,
           const Strides &out_strides) override;

  void sub(const DeviceBuffer &a, size_t a_offset, const Shape &a_shape,
           const Strides &a_strides, const DeviceBuffer &b, size_t b_offset,
           const Shape &b_shape, const Strides &b_strides, DeviceBuffer &out,
           size_t out_offset, const Shape &out_shape,
           const Strides &out_strides) override;

  void mul(const DeviceBuffer &a, size_t a_offset, const Shape &a_shape,
           const Strides &a_strides, const DeviceBuffer &b, size_t b_offset,
           const Shape &b_shape, const Strides &b_strides, DeviceBuffer &out,
           size_t out_offset, const Shape &out_shape,
           const Strides &out_strides) override;

  void div(const DeviceBuffer &a, size_t a_offset, const Shape &a_shape,
           const Strides &a_strides, const DeviceBuffer &b, size_t b_offset,
           const Shape &b_shape, const Strides &b_strides, DeviceBuffer &out,
           size_t out_offset, const Shape &out_shape,
           const Strides &out_strides) override;

  void matmul(const DeviceBuffer &a, size_t a_offset, const Shape &a_shape,
              const Strides &a_strides, const DeviceBuffer &b, size_t b_offset,
              const Shape &b_shape, const Strides &b_strides, DeviceBuffer &out,
              size_t out_offset, const Shape &out_shape,
              const Strides &out_strides) override;

  void sum(const DeviceBuffer &input, size_t input_offset,
           const Shape &input_shape, const Strides &input_strides,
           DeviceBuffer &output, size_t output_offset,
           const Shape &output_shape, const Strides &output_strides,
           const std::vector<size_t> &axes) override;

  void relu(const DeviceBuffer &in, size_t in_offset, const Shape &in_shape,
            const Strides &in_strides, DeviceBuffer &out, size_t out_offset,
            const Shape &out_shape, const Strides &out_strides) override;

  void relu_backward(const DeviceBuffer &in, size_t in_offset,
                     const Shape &in_shape, const Strides &in_strides,
                     const DeviceBuffer &grad_out, size_t grad_out_offset,
                     const Shape &grad_out_shape,
                     const Strides &grad_out_strides, DeviceBuffer &grad_in,
                     size_t grad_in_offset, const Shape &grad_in_shape,
                     const Strides &grad_in_strides) override;

  void sigmoid(const DeviceBuffer &in, size_t in_offset, const Shape &in_shape,
               const Strides &in_strides, DeviceBuffer &out, size_t out_offset,
               const Shape &out_shape, const Strides &out_strides) override;

  void sigmoid_backward(const DeviceBuffer &out, size_t out_offset,
                        const Shape &out_shape, const Strides &out_strides,
                        const DeviceBuffer &grad_out, size_t grad_out_offset,
                        const Shape &grad_out_shape,
                        const Strides &grad_out_strides, DeviceBuffer &grad_in,
                        size_t grad_in_offset, const Shape &grad_in_shape,
                        const Strides &grad_in_strides) override;

  void tanh(const DeviceBuffer &in, size_t in_offset, const Shape &in_shape,
            const Strides &in_strides, DeviceBuffer &out, size_t out_offset,
            const Shape &out_shape, const Strides &out_strides) override;

  void tanh_backward(const DeviceBuffer &out, size_t out_offset,
                     const Shape &out_shape, const Strides &out_strides,
                     const DeviceBuffer &grad_out, size_t grad_out_offset,
                     const Shape &grad_out_shape,
                     const Strides &grad_out_strides, DeviceBuffer &grad_in,
                     size_t grad_in_offset, const Shape &grad_in_shape,
                     const Strides &grad_in_strides) override;

  void softmax(const DeviceBuffer &in, size_t in_offset, const Shape &in_shape,
               const Strides &in_strides, DeviceBuffer &out, size_t out_offset,
               const Shape &out_shape, const Strides &out_strides,
               size_t axis) override;

  void softmax_backward(const DeviceBuffer &out, size_t out_offset,
                        const Shape &out_shape, const Strides &out_strides,
                        const DeviceBuffer &grad_out, size_t grad_out_offset,
                        const Shape &grad_out_shape,
                        const Strides &grad_out_strides, DeviceBuffer &grad_in,
                        size_t grad_in_offset, const Shape &grad_in_shape,
                        const Strides &grad_in_strides, size_t axis) override;

  void sqrt(const DeviceBuffer &in, size_t in_offset, const Shape &in_shape,
            const Strides &in_strides, DeviceBuffer &out, size_t out_offset,
            const Shape &out_shape, const Strides &out_strides) override;

  void sqrt_backward(const DeviceBuffer &out, size_t out_offset,
                     const Shape &out_shape, const Strides &out_strides,
                     const DeviceBuffer &grad_out, size_t grad_out_offset,
                     const Shape &grad_out_shape,
                     const Strides &grad_out_strides, DeviceBuffer &grad_in,
                     size_t grad_in_offset, const Shape &grad_in_shape,
                     const Strides &grad_in_strides) override;

  void log(const DeviceBuffer &in, size_t in_offset, const Shape &in_shape,
           const Strides &in_strides, DeviceBuffer &out, size_t out_offset,
           const Shape &out_shape, const Strides &out_strides) override;

  void log_backward(const DeviceBuffer &in, size_t in_offset,
                    const Shape &in_shape, const Strides &in_strides,
                    const DeviceBuffer &grad_out, size_t grad_out_offset,
                    const Shape &grad_out_shape,
                    const Strides &grad_out_strides, DeviceBuffer &grad_in,
                    size_t grad_in_offset, const Shape &grad_in_shape,
                    const Strides &grad_in_strides) override;

  void conv2d(const DeviceBuffer &in, size_t in_offset, const Shape &in_shape,
              const Strides &in_strides, const DeviceBuffer &weight,
              size_t weight_offset, const Shape &weight_shape,
              const Strides &weight_strides, const DeviceBuffer &bias,
              size_t bias_offset, const Shape &bias_shape,
              const Strides &bias_strides, DeviceBuffer &out,
              size_t out_offset, const Shape &out_shape,
              const Strides &out_strides, size_t padding,
              size_t stride) override;

  void conv2d_backward(
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
      const Strides &grad_bias_strides, size_t padding,
      size_t stride) override;

  void maxpool2d(const DeviceBuffer &in, size_t in_offset,
                 const Shape &in_shape, const Strides &in_strides,
                 DeviceBuffer &out, size_t out_offset,
                 const Shape &out_shape, const Strides &out_strides,
                 size_t pool_h, size_t pool_w, size_t stride) override;

  void maxpool2d_backward(
      const DeviceBuffer &in, size_t in_offset, const Shape &in_shape,
      const Strides &in_strides, const DeviceBuffer &out, size_t out_offset,
      const Shape &out_shape, const Strides &out_strides,
      const DeviceBuffer &grad_out, size_t grad_out_offset,
      const Shape &grad_out_shape, const Strides &grad_out_strides,
      DeviceBuffer &grad_in, size_t grad_in_offset, const Shape &grad_in_shape,
      const Strides &grad_in_strides, size_t pool_h, size_t pool_w,
      size_t stride) override;

  void embedding(const DeviceBuffer &weight, size_t weight_offset,
                 const Shape &weight_shape, const Strides &weight_strides,
                 const DeviceBuffer &indices, size_t indices_offset,
                 const Shape &indices_shape, const Strides &indices_strides,
                 DeviceBuffer &out, size_t out_offset, const Shape &out_shape,
                 const Strides &out_strides) override;

  void embedding_backward(
      const DeviceBuffer &grad_out, size_t grad_out_offset,
      const Shape &grad_out_shape, const Strides &grad_out_strides,
      const DeviceBuffer &indices, size_t indices_offset,
      const Shape &indices_shape, const Strides &indices_strides,
      DeviceBuffer &grad_weight, size_t grad_weight_offset,
      const Shape &grad_weight_shape,
      const Strides &grad_weight_strides) override;

  void slice_backward(const DeviceBuffer &grad_slice, size_t gs_offset,
                      const Shape &gs_shape, const Strides &gs_strides,
                      DeviceBuffer &grad_parent, size_t gp_offset,
                      const Shape &gp_shape, const Strides &gp_strides,
                      size_t axis, size_t index) override;

private:
#ifdef WITH_DML
  ComPtr<ID3D12Device> d3d12Device_;
  ComPtr<IDMLDevice> dmlDevice_;
  ComPtr<ID3D12CommandQueue> commandQueue_;
  ComPtr<ID3D12CommandAllocator> commandAllocator_;
  ComPtr<ID3D12GraphicsCommandList> commandList_;
  ComPtr<IDMLCommandRecorder> commandRecorder_;
#endif
};

#endif // DML_BACKEND_HPP
