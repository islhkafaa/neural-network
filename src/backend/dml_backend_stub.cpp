#include "backend/dml_backend.hpp"
#include <stdexcept>

DmlBuffer::DmlBuffer(size_t size_bytes, void *) : size_(size_bytes) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

DmlBuffer::~DmlBuffer() {}

void *DmlBuffer::data() noexcept { return nullptr; }
const void *DmlBuffer::data() const noexcept { return nullptr; }

DmlBackend::DmlBackend() {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

DmlBackend::~DmlBackend() {}

std::unique_ptr<DeviceBuffer> DmlBackend::allocate(size_t bytes) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::copy_host_to_device(const float *, DeviceBuffer &, size_t) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::copy_to_device(const float *, DeviceBuffer &, size_t, size_t) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::copy_device_to_host(const DeviceBuffer &, float *, size_t) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::fill(DeviceBuffer &, float) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::add(const DeviceBuffer &, size_t, const Shape &, const Strides &,
                     const DeviceBuffer &, size_t, const Shape &, const Strides &,
                     DeviceBuffer &, size_t, const Shape &, const Strides &) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::sub(const DeviceBuffer &, size_t, const Shape &, const Strides &,
                     const DeviceBuffer &, size_t, const Shape &, const Strides &,
                     DeviceBuffer &, size_t, const Shape &, const Strides &) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::mul(const DeviceBuffer &, size_t, const Shape &, const Strides &,
                     const DeviceBuffer &, size_t, const Shape &, const Strides &,
                     DeviceBuffer &, size_t, const Shape &, const Strides &) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::div(const DeviceBuffer &, size_t, const Shape &, const Strides &,
                     const DeviceBuffer &, size_t, const Shape &, const Strides &,
                     DeviceBuffer &, size_t, const Shape &, const Strides &) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::matmul(const DeviceBuffer &, size_t, const Shape &, const Strides &,
                        const DeviceBuffer &, size_t, const Shape &, const Strides &,
                        DeviceBuffer &, size_t, const Shape &, const Strides &) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::sum(const DeviceBuffer &, size_t, const Shape &, const Strides &,
                     DeviceBuffer &, size_t, const Shape &, const Strides &,
                     const std::vector<size_t> &) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::relu(const DeviceBuffer &, size_t, const Shape &, const Strides &,
                      DeviceBuffer &, size_t, const Shape &, const Strides &) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::relu_backward(const DeviceBuffer &, size_t, const Shape &,
                               const Strides &, const DeviceBuffer &, size_t,
                               const Shape &, const Strides &, DeviceBuffer &,
                               size_t, const Shape &, const Strides &) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::sigmoid(const DeviceBuffer &, size_t, const Shape &, const Strides &,
                         DeviceBuffer &, size_t, const Shape &, const Strides &) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::sigmoid_backward(const DeviceBuffer &, size_t, const Shape &,
                                  const Strides &, const DeviceBuffer &, size_t,
                                  const Shape &, const Strides &, DeviceBuffer &,
                                  size_t, const Shape &, const Strides &) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::tanh(const DeviceBuffer &, size_t, const Shape &, const Strides &,
                      DeviceBuffer &, size_t, const Shape &, const Strides &) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::tanh_backward(const DeviceBuffer &, size_t, const Shape &,
                               const Strides &, const DeviceBuffer &, size_t,
                               const Shape &, const Strides &, DeviceBuffer &,
                               size_t, const Shape &, const Strides &) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::softmax(const DeviceBuffer &, size_t, const Shape &, const Strides &,
                         DeviceBuffer &, size_t, const Shape &, const Strides &,
                         size_t) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::softmax_backward(const DeviceBuffer &, size_t, const Shape &,
                                  const Strides &, const DeviceBuffer &, size_t,
                                  const Shape &, const Strides &, DeviceBuffer &,
                                  size_t, const Shape &, const Strides &, size_t) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::sqrt(const DeviceBuffer &, size_t, const Shape &, const Strides &,
                      DeviceBuffer &, size_t, const Shape &, const Strides &) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::sqrt_backward(const DeviceBuffer &, size_t, const Shape &,
                               const Strides &, const DeviceBuffer &, size_t,
                               const Shape &, const Strides &, DeviceBuffer &,
                               size_t, const Shape &, const Strides &) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::log(const DeviceBuffer &, size_t, const Shape &, const Strides &,
                     DeviceBuffer &, size_t, const Shape &, const Strides &) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::log_backward(const DeviceBuffer &, size_t, const Shape &,
                              const Strides &, const DeviceBuffer &, size_t,
                              const Shape &, const Strides &, DeviceBuffer &,
                              size_t, const Shape &, const Strides &) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::conv2d(const DeviceBuffer &, size_t, const Shape &, const Strides &,
                        const DeviceBuffer &, size_t, const Shape &, const Strides &,
                        const DeviceBuffer &, size_t, const Shape &, const Strides &,
                        DeviceBuffer &, size_t, const Shape &, const Strides &,
                        size_t, size_t) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::conv2d_backward(
    const DeviceBuffer &, size_t, const Shape &, const Strides &,
    const DeviceBuffer &, size_t, const Shape &, const Strides &,
    const DeviceBuffer &, size_t, const Shape &, const Strides &, DeviceBuffer &,
    size_t, const Shape &, const Strides &, DeviceBuffer &, size_t, const Shape &,
    const Strides &, DeviceBuffer &, size_t, const Shape &, const Strides &,
    size_t, size_t) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::maxpool2d(const DeviceBuffer &, size_t, const Shape &, const Strides &,
                           DeviceBuffer &, size_t, const Shape &, const Strides &,
                           size_t, size_t, size_t) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::maxpool2d_backward(
    const DeviceBuffer &, size_t, const Shape &, const Strides &,
    const DeviceBuffer &, size_t, const Shape &, const Strides &,
    const DeviceBuffer &, size_t, const Shape &, const Strides &, DeviceBuffer &,
    size_t, const Shape &, const Strides &, size_t, size_t, size_t) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::embedding(const DeviceBuffer &, size_t, const Shape &,
                           const Strides &, const DeviceBuffer &, size_t,
                           const Shape &, const Strides &, DeviceBuffer &,
                           size_t, const Shape &, const Strides &) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::embedding_backward(
    const DeviceBuffer &, size_t, const Shape &, const Strides &,
    const DeviceBuffer &, size_t, const Shape &, const Strides &, DeviceBuffer &,
    size_t, const Shape &, const Strides &) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}

void DmlBackend::slice_backward(const DeviceBuffer &, size_t, const Shape &,
                                const Strides &, DeviceBuffer &, size_t,
                                const Shape &, const Strides &, size_t, size_t) {
  throw std::runtime_error("DirectML is not enabled in this build.");
}
