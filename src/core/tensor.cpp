#include "core/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

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

static Shape broadcast_shapes(const Shape &a, const Shape &b) {
  Shape out;
  size_t R_a = a.size();
  size_t R_b = b.size();
  size_t R_out = std::max(R_a, R_b);
  out.resize(R_out);
  for (size_t i = 0; i < R_out; ++i) {
    size_t d_a = (i < R_a) ? a[R_a - 1 - i] : 1;
    size_t d_b = (i < R_b) ? b[R_b - 1 - i] : 1;
    if (d_a != d_b && d_a != 1 && d_b != 1) {
      throw std::runtime_error("Incompatible shapes for broadcasting.");
    }
    out[R_out - 1 - i] = std::max(d_a, d_b);
  }
  return out;
}

static Strides broadcast_strides(const Shape &orig_shape,
                                 const Strides &orig_strides,
                                 const Shape &target_shape) {
  size_t R_orig = orig_shape.size();
  size_t R_tar = target_shape.size();
  Strides tar_strides(R_tar, 0);

  if (R_tar >= R_orig) {
    size_t diff = R_tar - R_orig;
    for (size_t i = 0; i < R_tar; ++i) {
      if (i < diff) {
        tar_strides[i] = 0;
      } else {
        size_t dim_orig = i - diff;
        if (orig_shape[dim_orig] == 1 && target_shape[i] > 1) {
          tar_strides[i] = 0;
        } else {
          tar_strides[i] = orig_strides[dim_orig];
        }
      }
    }
  } else {
    size_t diff = R_orig - R_tar;
    for (size_t i = 0; i < R_tar; ++i) {
      size_t dim_orig = i + diff;
      if (orig_shape[dim_orig] == 1 && target_shape[i] > 1) {
        tar_strides[i] = 0;
      } else {
        tar_strides[i] = orig_strides[dim_orig];
      }
    }
  }
  return tar_strides;
}

std::shared_ptr<Tensor>
reduce_grad_for_broadcasting(const std::shared_ptr<Tensor> &grad,
                             const Shape &target_shape) {
  if (grad->shape() == target_shape) {
    return grad;
  }
  std::vector<size_t> axes;
  Shape grad_shape = grad->shape();
  size_t R_grad = grad_shape.size();
  size_t R_tar = target_shape.size();

  for (size_t i = 0; i < R_grad; ++i) {
    size_t dim = R_grad - 1 - i;
    if (i >= R_tar) {
      axes.push_back(dim);
    } else {
      size_t tar_dim = R_tar - 1 - i;
      if (target_shape[tar_dim] == 1 && grad_shape[dim] > 1) {
        axes.push_back(dim);
      }
    }
  }
  std::reverse(axes.begin(), axes.end());

  if (axes.empty()) {
    return grad;
  }

  auto reduced = grad->sum(axes, true);
  if (reduced->shape() != target_shape) {
    return std::make_shared<Tensor>(
        target_shape,
        broadcast_strides(reduced->shape(), reduced->strides(), target_shape),
        reduced->offset(), reduced->storage(), reduced->backend(), reduced->dtype());
  }
  return reduced;
}

Tensor::Tensor(const Shape &shape, ExecutionBackend *backend, DataType dtype)
    : shape_(shape), strides_(compute_strides(shape)), offset_(0),
      backend_(backend ? backend : default_backend()), dtype_(dtype) {
  storage_ = backend_->allocate(count_elements(shape_) * dtype_size(dtype_));
  storage_->set_dtype(dtype_);
}

Tensor::Tensor(const Shape &shape, const std::vector<float> &host_data,
               ExecutionBackend *backend, DataType dtype)
    : shape_(shape), strides_(compute_strides(shape)), offset_(0),
      backend_(backend ? backend : default_backend()), dtype_(dtype) {
  if (host_data.size() != count_elements(shape_)) {
    throw std::runtime_error("Data size mismatch during tensor construction.");
  }
  storage_ = backend_->allocate(count_elements(shape_) * dtype_size(dtype_));
  storage_->set_dtype(dtype_);
  if (dtype_ == DataType::FP32) {
    backend_->copy_host_to_device(host_data.data(), *storage_,
                                  host_data.size() * sizeof(float));
  } else {
    std::vector<uint16_t> half_data(host_data.size());
    if (dtype_ == DataType::FP16) {
      for (size_t i = 0; i < host_data.size(); ++i) {
        half_data[i] = float_to_fp16(host_data[i]);
      }
    } else {
      for (size_t i = 0; i < host_data.size(); ++i) {
        half_data[i] = float_to_bfloat16(host_data[i]);
      }
    }
    backend_->copy_host_to_device(reinterpret_cast<const float *>(half_data.data()), *storage_,
                                  host_data.size() * sizeof(uint16_t));
  }
}

Tensor::Tensor(const Shape &shape, const Strides &strides, size_t offset,
               std::shared_ptr<DeviceBuffer> storage, ExecutionBackend *backend, DataType dtype)
    : shape_(shape), strides_(strides), offset_(offset),
      storage_(std::move(storage)),
      backend_(backend ? backend : default_backend()), dtype_(dtype) {
  if (storage_) {
    storage_->set_dtype(dtype_);
  }
}

std::vector<float> Tensor::to_host() const {
  size_t size = count_elements(shape_);
  std::vector<float> host_data(size);
  if (dtype_ == DataType::FP32) {
    std::vector<float> full_buf(storage_->size() / sizeof(float));
    backend_->copy_device_to_host(*storage_, full_buf.data(), storage_->size());
    for (size_t i = 0; i < size; ++i) {
      size_t idx = get_input_offset(i, shape_, shape_, strides_);
      host_data[i] = full_buf[offset_ + idx];
    }
  } else {
    std::vector<uint16_t> full_buf(storage_->size() / sizeof(uint16_t));
    backend_->copy_device_to_host(*storage_, reinterpret_cast<float *>(full_buf.data()), storage_->size());
    if (dtype_ == DataType::FP16) {
      for (size_t i = 0; i < size; ++i) {
        size_t idx = get_input_offset(i, shape_, shape_, strides_);
        host_data[i] = fp16_to_float(full_buf[offset_ + idx]);
      }
    } else {
      for (size_t i = 0; i < size; ++i) {
        size_t idx = get_input_offset(i, shape_, shape_, strides_);
        host_data[i] = bfloat16_to_float(full_buf[offset_ + idx]);
      }
    }
  }
  return host_data;
}

void Tensor::fill(float value) {
  if (is_contiguous(shape_, strides_)) {
    if (dtype_ == DataType::FP32 || value == 0.0f) {
      backend_->fill(*storage_, value);
    } else {
      uint16_t half_val = (dtype_ == DataType::FP16) ? float_to_fp16(value) : float_to_bfloat16(value);
      std::vector<uint16_t> half_data(count_elements(shape_), half_val);
      backend_->copy_host_to_device(reinterpret_cast<const float *>(half_data.data()), *storage_,
                                    half_data.size() * sizeof(uint16_t));
    }
  } else {
    if (dtype_ == DataType::FP32) {
      std::vector<float> full_buf(storage_->size() / sizeof(float));
      backend_->copy_device_to_host(*storage_, full_buf.data(), storage_->size());
      size_t size = count_elements(shape_);
      for (size_t i = 0; i < size; ++i) {
        size_t idx = get_input_offset(i, shape_, shape_, strides_);
        full_buf[offset_ + idx] = value;
      }
      backend_->copy_host_to_device(full_buf.data(), *storage_, storage_->size());
    } else {
      std::vector<uint16_t> full_buf(storage_->size() / sizeof(uint16_t));
      backend_->copy_device_to_host(*storage_, reinterpret_cast<float *>(full_buf.data()), storage_->size());
      uint16_t half_val = (dtype_ == DataType::FP16) ? float_to_fp16(value) : float_to_bfloat16(value);
      size_t size = count_elements(shape_);
      for (size_t i = 0; i < size; ++i) {
        size_t idx = get_input_offset(i, shape_, shape_, strides_);
        full_buf[offset_ + idx] = half_val;
      }
      backend_->copy_host_to_device(reinterpret_cast<const float *>(full_buf.data()), *storage_, storage_->size());
    }
  }
}

void Tensor::copy_from_host(const std::vector<float> &host_data) {
  if (host_data.size() != count_elements(shape_)) {
    throw std::runtime_error("Size mismatch in copy_from_host");
  }
  if (dtype_ == DataType::FP32) {
    if (is_contiguous(shape_, strides_)) {
      backend_->copy_host_to_device(host_data.data(), *storage_,
                                    host_data.size() * sizeof(float));
    } else {
      std::vector<float> full_buf(storage_->size() / sizeof(float));
      backend_->copy_device_to_host(*storage_, full_buf.data(), storage_->size());
      size_t size = count_elements(shape_);
      for (size_t i = 0; i < size; ++i) {
        size_t idx = get_input_offset(i, shape_, shape_, strides_);
        full_buf[offset_ + idx] = host_data[i];
      }
      backend_->copy_host_to_device(full_buf.data(), *storage_, storage_->size());
    }
  } else {
    std::vector<uint16_t> half_data(host_data.size());
    if (dtype_ == DataType::FP16) {
      for (size_t i = 0; i < host_data.size(); ++i) {
        half_data[i] = float_to_fp16(host_data[i]);
      }
    } else {
      for (size_t i = 0; i < host_data.size(); ++i) {
        half_data[i] = float_to_bfloat16(host_data[i]);
      }
    }
    if (is_contiguous(shape_, strides_)) {
      backend_->copy_host_to_device(reinterpret_cast<const float *>(half_data.data()), *storage_,
                                    half_data.size() * sizeof(uint16_t));
    } else {
      std::vector<uint16_t> full_buf(storage_->size() / sizeof(uint16_t));
      backend_->copy_device_to_host(*storage_, reinterpret_cast<float *>(full_buf.data()), storage_->size());
      size_t size = count_elements(shape_);
      for (size_t i = 0; i < size; ++i) {
        size_t idx = get_input_offset(i, shape_, shape_, strides_);
        full_buf[offset_ + idx] = half_data[i];
      }
      backend_->copy_host_to_device(reinterpret_cast<const float *>(full_buf.data()), *storage_, storage_->size());
    }
  }
}

std::shared_ptr<Tensor> Tensor::add(const std::shared_ptr<Tensor> &other) {
  Shape out_shape = broadcast_shapes(shape_, other->shape());
  auto out_storage =
      backend_->allocate(count_elements(out_shape) * dtype_size(dtype_));
  out_storage->set_dtype(dtype_);
  Strides out_strides = compute_strides(out_shape);

  backend_->add(*storage_, offset_, shape_, strides_, *other->storage(),
                other->offset(), other->shape(), other->strides(), *out_storage,
                0, out_shape, out_strides);

  auto out = std::make_shared<Tensor>(out_shape, out_strides, 0,
                                      std::move(out_storage), backend_, dtype_);
  out->requires_grad_ = requires_grad_ || other->requires_grad_;
  if (out->requires_grad_) {
    out->inputs_ = {shared_from_this(), other};
    out->backward_ = [self = shared_from_this(), other, out]() {
      if (self->requires_grad_) {
        self->accumulate_grad(
            reduce_grad_for_broadcasting(out->grad(), self->shape()));
      }
      if (other->requires_grad_) {
        other->accumulate_grad(
            reduce_grad_for_broadcasting(out->grad(), other->shape()));
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> Tensor::sub(const std::shared_ptr<Tensor> &other) {
  Shape out_shape = broadcast_shapes(shape_, other->shape());
  auto out_storage =
      backend_->allocate(count_elements(out_shape) * dtype_size(dtype_));
  out_storage->set_dtype(dtype_);
  Strides out_strides = compute_strides(out_shape);

  backend_->sub(*storage_, offset_, shape_, strides_, *other->storage(),
                other->offset(), other->shape(), other->strides(), *out_storage,
                0, out_shape, out_strides);

  auto out = std::make_shared<Tensor>(out_shape, out_strides, 0,
                                      std::move(out_storage), backend_, dtype_);
  out->requires_grad_ = requires_grad_ || other->requires_grad_;
  if (out->requires_grad_) {
    out->inputs_ = {shared_from_this(), other};
    out->backward_ = [self = shared_from_this(), other, out]() {
      if (self->requires_grad_) {
        self->accumulate_grad(
            reduce_grad_for_broadcasting(out->grad(), self->shape()));
      }
      if (other->requires_grad_) {
        other->accumulate_grad(
            reduce_grad_for_broadcasting(-out->grad(), other->shape()));
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> Tensor::mul(const std::shared_ptr<Tensor> &other) {
  Shape out_shape = broadcast_shapes(shape_, other->shape());
  auto out_storage =
      backend_->allocate(count_elements(out_shape) * dtype_size(dtype_));
  out_storage->set_dtype(dtype_);
  Strides out_strides = compute_strides(out_shape);

  backend_->mul(*storage_, offset_, shape_, strides_, *other->storage(),
                other->offset(), other->shape(), other->strides(), *out_storage,
                0, out_shape, out_strides);

  auto out = std::make_shared<Tensor>(out_shape, out_strides, 0,
                                      std::move(out_storage), backend_, dtype_);
  out->requires_grad_ = requires_grad_ || other->requires_grad_;
  if (out->requires_grad_) {
    out->inputs_ = {shared_from_this(), other};
    out->backward_ = [self = shared_from_this(), other, out]() {
      if (self->requires_grad_) {
        self->accumulate_grad(
            reduce_grad_for_broadcasting(out->grad() * other, self->shape()));
      }
      if (other->requires_grad_) {
        other->accumulate_grad(
            reduce_grad_for_broadcasting(out->grad() * self, other->shape()));
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> Tensor::div(const std::shared_ptr<Tensor> &other) {
  Shape out_shape = broadcast_shapes(shape_, other->shape());
  auto out_storage =
      backend_->allocate(count_elements(out_shape) * dtype_size(dtype_));
  out_storage->set_dtype(dtype_);
  Strides out_strides = compute_strides(out_shape);

  backend_->div(*storage_, offset_, shape_, strides_, *other->storage(),
                other->offset(), other->shape(), other->strides(), *out_storage,
                0, out_shape, out_strides);

  auto out = std::make_shared<Tensor>(out_shape, out_strides, 0,
                                      std::move(out_storage), backend_, dtype_);
  out->requires_grad_ = requires_grad_ || other->requires_grad_;
  if (out->requires_grad_) {
    out->inputs_ = {shared_from_this(), other};
    out->backward_ = [self = shared_from_this(), other, out]() {
      if (self->requires_grad_) {
        self->accumulate_grad(
            reduce_grad_for_broadcasting(out->grad() / other, self->shape()));
      }
      if (other->requires_grad_) {
        other->accumulate_grad(reduce_grad_for_broadcasting(
            out->grad() * (-self / (other * other)), other->shape()));
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> Tensor::matmul(const std::shared_ptr<Tensor> &other) {
  if (shape_.size() != 2 || other->shape().size() != 2) {
    throw std::runtime_error("Matmul only supports 2D matrices.");
  }
  size_t M = shape_[0];
  size_t K = shape_[1];
  size_t N = other->shape()[1];

  if (other->shape()[0] != K) {
    throw std::runtime_error("Incompatible shapes for matmul.");
  }

  Shape out_shape = {M, N};
  auto out_storage = backend_->allocate(M * N * dtype_size(dtype_));
  out_storage->set_dtype(dtype_);
  Strides out_strides = compute_strides(out_shape);

  backend_->matmul(*storage_, offset_, shape_, strides_, *other->storage(),
                   other->offset(), other->shape(), other->strides(),
                   *out_storage, 0, out_shape, out_strides);

  auto out = std::make_shared<Tensor>(out_shape, out_strides, 0,
                                      std::move(out_storage), backend_, dtype_);
  out->requires_grad_ = requires_grad_ || other->requires_grad_;
  if (out->requires_grad_) {
    out->inputs_ = {shared_from_this(), other};
    out->backward_ = [self = shared_from_this(), other, out]() {
      if (self->requires_grad_) {
        self->accumulate_grad(out->grad()->matmul(other->transpose()));
      }
      if (other->requires_grad_) {
        other->accumulate_grad(self->transpose()->matmul(out->grad()));
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> Tensor::sum(const std::vector<size_t> &axes,
                                    bool keep_dims) {
  for (size_t axis : axes) {
    if (axis >= shape_.size()) {
      throw std::runtime_error("Axis out of range for sum.");
    }
  }

  Shape out_shape;
  if (keep_dims) {
    out_shape = shape_;
    for (size_t axis : axes) {
      out_shape[axis] = 1;
    }
  } else {
    for (size_t i = 0; i < shape_.size(); ++i) {
      bool is_summed = false;
      for (size_t axis : axes) {
        if (axis == i) {
          is_summed = true;
          break;
        }
      }
      if (!is_summed) {
        out_shape.push_back(shape_[i]);
      }
    }
  }

  auto out_storage =
      backend_->allocate(count_elements(out_shape) * dtype_size(dtype_));
  out_storage->set_dtype(dtype_);
  Strides out_strides = compute_strides(out_shape);

  backend_->sum(*storage_, offset_, shape_, strides_, *out_storage, 0,
                out_shape, out_strides, axes);

  auto out = std::make_shared<Tensor>(out_shape, out_strides, 0,
                                      std::move(out_storage), backend_, dtype_);
  out->requires_grad_ = requires_grad_;
  if (requires_grad_) {
    out->inputs_ = {shared_from_this()};
    out->backward_ = [self = shared_from_this(), out, keep_dims]() {
      if (self->requires_grad_) {
        auto incoming = out->grad();
        if (!keep_dims) {
          Shape expanded_shape = self->shape();
          incoming = std::make_shared<Tensor>(
              expanded_shape,
              broadcast_strides(out->grad()->shape(), out->grad()->strides(),
                                expanded_shape),
              out->grad()->offset(), out->grad()->storage(), out->backend(), out->dtype());
        }
        self->accumulate_grad(incoming);
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> Tensor::transpose() {
  if (shape_.size() != 2) {
    throw std::runtime_error("Transpose helper only supported for 2D tensors.");
  }
  return transpose(0, 1);
}

std::shared_ptr<Tensor> Tensor::transpose(size_t dim1, size_t dim2) {
  if (dim1 >= shape_.size() || dim2 >= shape_.size()) {
    throw std::runtime_error("Transpose dimensions out of range.");
  }
  Shape trans_shape = shape_;
  Strides trans_strides = strides_;
  std::swap(trans_shape[dim1], trans_shape[dim2]);
  std::swap(trans_strides[dim1], trans_strides[dim2]);

  auto out = std::make_shared<Tensor>(trans_shape, trans_strides, offset_,
                                      storage_, backend_, dtype_);
  out->requires_grad_ = requires_grad_;
  if (requires_grad_) {
    out->inputs_ = {shared_from_this()};
    out->backward_ = [self = shared_from_this(), out, dim1, dim2]() {
      if (self->requires_grad_) {
        self->accumulate_grad(out->grad()->transpose(dim1, dim2));
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> Tensor::reshape(const Shape &new_shape) {
  if (count_elements(shape_) != count_elements(new_shape)) {
    throw std::runtime_error("Cannot reshape tensor: element count mismatch.");
  }

  std::shared_ptr<Tensor> base = shared_from_this();
  if (offset_ != 0 || !is_contiguous(shape_, strides_)) {
    auto contiguous_tensor =
        std::make_shared<Tensor>(shape_, to_host(), backend_, dtype_);
    base = contiguous_tensor;
  }

  auto out_strides = compute_strides(new_shape);
  auto out = std::make_shared<Tensor>(new_shape, out_strides, base->offset(),
                                      base->storage(), backend_, base->dtype());

  out->requires_grad_ = requires_grad_;
  if (out->requires_grad_) {
    out->inputs_ = {shared_from_this()};
    out->backward_ = [self = shared_from_this(), out]() {
      if (self->requires_grad_) {
        self->accumulate_grad(out->grad()->reshape(self->shape()));
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> Tensor::bmm(const std::shared_ptr<Tensor> &other) {
  if (shape_.size() != 4 || other->shape().size() != 4) {
    throw std::runtime_error("bmm only supports 4D tensors currently.");
  }
  size_t batch_size = shape_[0];
  size_t num_heads = shape_[1];
  size_t K = shape_[3];

  if (other->shape()[0] != batch_size || other->shape()[1] != num_heads ||
      other->shape()[2] != K) {
    throw std::runtime_error("Incompatible shapes for bmm.");
  }

  std::vector<std::shared_ptr<Tensor>> batch_tensors;
  for (size_t b = 0; b < batch_size; ++b) {
    auto self_b = slice(0, b);
    auto other_b = other->slice(0, b);

    std::vector<std::shared_ptr<Tensor>> head_tensors;
    for (size_t h = 0; h < num_heads; ++h) {
      auto self_bh = self_b->slice(0, h);
      auto other_bh = other_b->slice(0, h);

      head_tensors.push_back(self_bh->matmul(other_bh));
    }
    batch_tensors.push_back(Tensor::stack(head_tensors));
  }
  return Tensor::stack(batch_tensors);
}

std::shared_ptr<Tensor> Tensor::relu() {
  auto out_storage = backend_->allocate(count_elements(shape_) * dtype_size(dtype_));
  out_storage->set_dtype(dtype_);
  Strides out_strides = compute_strides(shape_);
  backend_->relu(*storage_, offset_, shape_, strides_, *out_storage, 0, shape_,
                 out_strides);

  auto out = std::make_shared<Tensor>(shape_, out_strides, 0,
                                      std::move(out_storage), backend_, dtype_);
  out->requires_grad_ = requires_grad_;
  if (requires_grad_) {
    out->inputs_ = {shared_from_this()};
    out->backward_ = [self = shared_from_this(), out]() {
      if (self->requires_grad_) {
        auto grad_in_storage = self->backend()->allocate(
            count_elements(self->shape()) * dtype_size(self->dtype()));
        grad_in_storage->set_dtype(self->dtype());
        Strides gi_strides = compute_strides(self->shape());
        self->backend()->relu_backward(
            *self->storage(), self->offset(), self->shape(), self->strides(),
            *out->grad()->storage(), out->grad()->offset(),
            out->grad()->shape(), out->grad()->strides(), *grad_in_storage, 0,
            self->shape(), gi_strides);
        auto grad_in = std::make_shared<Tensor>(self->shape(), gi_strides, 0,
                                                std::move(grad_in_storage),
                                                self->backend(), self->dtype());
        self->accumulate_grad(grad_in);
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> Tensor::sigmoid() {
  auto out_storage = backend_->allocate(count_elements(shape_) * dtype_size(dtype_));
  out_storage->set_dtype(dtype_);
  Strides out_strides = compute_strides(shape_);
  backend_->sigmoid(*storage_, offset_, shape_, strides_, *out_storage, 0,
                    shape_, out_strides);

  auto out = std::make_shared<Tensor>(shape_, out_strides, 0,
                                      std::move(out_storage), backend_, dtype_);
  out->requires_grad_ = requires_grad_;
  if (requires_grad_) {
    out->inputs_ = {shared_from_this()};
    out->backward_ = [self = shared_from_this(), out]() {
      if (self->requires_grad_) {
        auto grad_in_storage = self->backend()->allocate(
            count_elements(self->shape()) * dtype_size(self->dtype()));
        grad_in_storage->set_dtype(self->dtype());
        Strides gi_strides = compute_strides(self->shape());
        self->backend()->sigmoid_backward(
            *out->storage(), out->offset(), out->shape(), out->strides(),
            *out->grad()->storage(), out->grad()->offset(),
            out->grad()->shape(), out->grad()->strides(), *grad_in_storage, 0,
            self->shape(), gi_strides);
        auto grad_in = std::make_shared<Tensor>(self->shape(), gi_strides, 0,
                                                std::move(grad_in_storage),
                                                self->backend(), self->dtype());
        self->accumulate_grad(grad_in);
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> Tensor::tanh() {
  auto out_storage = backend_->allocate(count_elements(shape_) * dtype_size(dtype_));
  out_storage->set_dtype(dtype_);
  Strides out_strides = compute_strides(shape_);
  backend_->tanh(*storage_, offset_, shape_, strides_, *out_storage, 0, shape_,
                 out_strides);

  auto out = std::make_shared<Tensor>(shape_, out_strides, 0,
                                      std::move(out_storage), backend_, dtype_);
  out->requires_grad_ = requires_grad_;
  if (requires_grad_) {
    out->inputs_ = {shared_from_this()};
    out->backward_ = [self = shared_from_this(), out]() {
      if (self->requires_grad_) {
        auto grad_in_storage = self->backend()->allocate(
            count_elements(self->shape()) * dtype_size(self->dtype()));
        grad_in_storage->set_dtype(self->dtype());
        Strides gi_strides = compute_strides(self->shape());
        self->backend()->tanh_backward(
            *out->storage(), out->offset(), out->shape(), out->strides(),
            *out->grad()->storage(), out->grad()->offset(),
            out->grad()->shape(), out->grad()->strides(), *grad_in_storage, 0,
            self->shape(), gi_strides);
        auto grad_in = std::make_shared<Tensor>(self->shape(), gi_strides, 0,
                                                std::move(grad_in_storage),
                                                self->backend(), self->dtype());
        self->accumulate_grad(grad_in);
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> Tensor::softmax(size_t axis) {
  if (axis >= shape_.size()) {
    throw std::runtime_error("Axis out of bounds for softmax.");
  }
  auto out_storage = backend_->allocate(count_elements(shape_) * dtype_size(dtype_));
  out_storage->set_dtype(dtype_);
  Strides out_strides = compute_strides(shape_);
  backend_->softmax(*storage_, offset_, shape_, strides_, *out_storage, 0,
                    shape_, out_strides, axis);

  auto out = std::make_shared<Tensor>(shape_, out_strides, 0,
                                      std::move(out_storage), backend_, dtype_);
  out->requires_grad_ = requires_grad_;
  if (requires_grad_) {
    out->inputs_ = {shared_from_this()};
    out->backward_ = [self = shared_from_this(), out, axis]() {
      if (self->requires_grad_) {
        auto grad_in_storage = self->backend()->allocate(
            count_elements(self->shape()) * dtype_size(self->dtype()));
        grad_in_storage->set_dtype(self->dtype());
        Strides gi_strides = compute_strides(self->shape());
        self->backend()->softmax_backward(
            *out->storage(), out->offset(), out->shape(), out->strides(),
            *out->grad()->storage(), out->grad()->offset(),
            out->grad()->shape(), out->grad()->strides(), *grad_in_storage, 0,
            self->shape(), gi_strides, axis);
        auto grad_in = std::make_shared<Tensor>(self->shape(), gi_strides, 0,
                                                std::move(grad_in_storage),
                                                self->backend(), self->dtype());
        self->accumulate_grad(grad_in);
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> Tensor::sqrt() {
  auto out_storage = backend_->allocate(count_elements(shape_) * dtype_size(dtype_));
  Strides out_strides = compute_strides(shape_);
  backend_->sqrt(*storage_, offset_, shape_, strides_, *out_storage, 0, shape_,
                 out_strides);

  auto out = std::make_shared<Tensor>(shape_, out_strides, 0,
                                      std::move(out_storage), backend_, dtype_);
  out->requires_grad_ = requires_grad_;
  if (requires_grad_) {
    out->inputs_ = {shared_from_this()};
    out->backward_ = [self = shared_from_this(), out]() {
      if (self->requires_grad_) {
        auto grad_in_storage = self->backend()->allocate(
            count_elements(self->shape()) * dtype_size(self->dtype()));
        Strides gi_strides = compute_strides(self->shape());
        self->backend()->sqrt_backward(
            *out->storage(), out->offset(), out->shape(), out->strides(),
            *out->grad()->storage(), out->grad()->offset(),
            out->grad()->shape(), out->grad()->strides(), *grad_in_storage, 0,
            self->shape(), gi_strides);
        auto grad_in = std::make_shared<Tensor>(self->shape(), gi_strides, 0,
                                                std::move(grad_in_storage),
                                                self->backend(), self->dtype());
        self->accumulate_grad(grad_in);
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> Tensor::log() {
  auto out_storage = backend_->allocate(count_elements(shape_) * dtype_size(dtype_));
  out_storage->set_dtype(dtype_);
  Strides out_strides = compute_strides(shape_);
  backend_->log(*storage_, offset_, shape_, strides_, *out_storage, 0, shape_,
                out_strides);

  auto out = std::make_shared<Tensor>(shape_, out_strides, 0,
                                      std::move(out_storage), backend_, dtype_);
  out->requires_grad_ = requires_grad_;
  if (requires_grad_) {
    out->inputs_ = {shared_from_this()};
    out->backward_ = [self = shared_from_this(), out]() {
      if (self->requires_grad_) {
        auto grad_in_storage = self->backend()->allocate(
            count_elements(self->shape()) * dtype_size(self->dtype()));
        grad_in_storage->set_dtype(self->dtype());
        Strides gi_strides = compute_strides(self->shape());
        self->backend()->log_backward(
            *self->storage(), self->offset(), self->shape(), self->strides(),
            *out->grad()->storage(), out->grad()->offset(),
            out->grad()->shape(), out->grad()->strides(), *grad_in_storage, 0,
            self->shape(), gi_strides);
        auto grad_in = std::make_shared<Tensor>(self->shape(), gi_strides, 0,
                                                std::move(grad_in_storage),
                                                self->backend(), self->dtype());
        self->accumulate_grad(grad_in);
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> Tensor::conv2d(const std::shared_ptr<Tensor> &weight,
                                       const std::shared_ptr<Tensor> &bias,
                                       size_t padding, size_t stride) {
  if (shape_.size() != 4 || weight->shape().size() != 4) {
    throw std::runtime_error("conv2d expects 4D input and weight tensors.");
  }
  size_t N = shape_[0];
  size_t H = shape_[2];
  size_t W = shape_[3];

  size_t C_out = weight->shape()[0];
  size_t K_h = weight->shape()[2];
  size_t K_w = weight->shape()[3];

  size_t H_out = (H - K_h + 2 * padding) / stride + 1;
  size_t W_out = (W - K_w + 2 * padding) / stride + 1;

  Shape out_shape = {N, C_out, H_out, W_out};
  auto out_storage =
      backend_->allocate(count_elements(out_shape) * dtype_size(dtype_));
  out_storage->set_dtype(dtype_);
  Strides out_strides = compute_strides(out_shape);

  backend_->conv2d(*storage_, offset_, shape_, strides_, *weight->storage(),
                   weight->offset(), weight->shape(), weight->strides(),
                   *bias->storage(), bias->offset(), bias->shape(),
                   bias->strides(), *out_storage, 0, out_shape, out_strides,
                   padding, stride);

  auto out = std::make_shared<Tensor>(out_shape, out_strides, 0,
                                      std::move(out_storage), backend_, dtype_);
  out->requires_grad_ =
      requires_grad_ || weight->requires_grad() || bias->requires_grad();

  if (out->requires_grad_) {
    out->inputs_ = {shared_from_this(), weight, bias};
    out->backward_ = [self = shared_from_this(), weight, bias, out, padding,
                      stride]() {
      auto grad_in_storage = self->backend()->allocate(
          count_elements(self->shape()) * dtype_size(self->dtype()));
      grad_in_storage->set_dtype(self->dtype());
      Strides gi_strides = compute_strides(self->shape());

      auto grad_w_storage = weight->backend()->allocate(
          count_elements(weight->shape()) * dtype_size(weight->dtype()));
      grad_w_storage->set_dtype(weight->dtype());
      Strides gw_strides = compute_strides(weight->shape());

      auto grad_b_storage = bias->backend()->allocate(
          count_elements(bias->shape()) * dtype_size(bias->dtype()));
      grad_b_storage->set_dtype(bias->dtype());
      Strides gb_strides = compute_strides(bias->shape());

      self->backend()->conv2d_backward(
          *self->storage(), self->offset(), self->shape(), self->strides(),
          *weight->storage(), weight->offset(), weight->shape(),
          weight->strides(), *out->grad()->storage(), out->grad()->offset(),
          out->grad()->shape(), out->grad()->strides(), *grad_in_storage, 0,
          self->shape(), gi_strides, *grad_w_storage, 0, weight->shape(),
          gw_strides, *grad_b_storage, 0, bias->shape(), gb_strides, padding,
          stride);

      if (self->requires_grad_) {
        auto grad_in = std::make_shared<Tensor>(self->shape(), gi_strides, 0,
                                                std::move(grad_in_storage),
                                                self->backend(), self->dtype());
        self->accumulate_grad(grad_in);
      }
      if (weight->requires_grad()) {
        auto grad_w = std::make_shared<Tensor>(weight->shape(), gw_strides, 0,
                                               std::move(grad_w_storage),
                                               weight->backend(), weight->dtype());
        weight->accumulate_grad(grad_w);
      }
      if (bias->requires_grad()) {
        auto grad_b = std::make_shared<Tensor>(bias->shape(), gb_strides, 0,
                                               std::move(grad_b_storage),
                                               bias->backend(), bias->dtype());
        bias->accumulate_grad(grad_b);
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor> Tensor::maxpool2d(size_t pool_h, size_t pool_w,
                                          size_t stride) {
  if (shape_.size() != 4) {
    throw std::runtime_error("maxpool2d expects 4D input tensor.");
  }
  size_t N = shape_[0];
  size_t C = shape_[1];
  size_t H = shape_[2];
  size_t W = shape_[3];

  size_t H_out = (H - pool_h) / stride + 1;
  size_t W_out = (W - pool_w) / stride + 1;

  Shape out_shape = {N, C, H_out, W_out};
  auto out_storage =
      backend_->allocate(count_elements(out_shape) * dtype_size(dtype_));
  out_storage->set_dtype(dtype_);
  Strides out_strides = compute_strides(out_shape);

  backend_->maxpool2d(*storage_, offset_, shape_, strides_, *out_storage, 0,
                      out_shape, out_strides, pool_h, pool_w, stride);

  auto out = std::make_shared<Tensor>(out_shape, out_strides, 0,
                                      std::move(out_storage), backend_, dtype_);
  out->requires_grad_ = requires_grad_;

  if (requires_grad_) {
    out->inputs_ = {shared_from_this()};
    out->backward_ = [self = shared_from_this(), out, pool_h, pool_w,
                      stride]() {
      if (self->requires_grad_) {
        auto grad_in_storage = self->backend()->allocate(
            count_elements(self->shape()) * dtype_size(self->dtype()));
        grad_in_storage->set_dtype(self->dtype());
        Strides gi_strides = compute_strides(self->shape());

        self->backend()->maxpool2d_backward(
            *self->storage(), self->offset(), self->shape(), self->strides(),
            *out->storage(), out->offset(), out->shape(), out->strides(),
            *out->grad()->storage(), out->grad()->offset(),
            out->grad()->shape(), out->grad()->strides(), *grad_in_storage, 0,
            self->shape(), gi_strides, pool_h, pool_w, stride);

        auto grad_in = std::make_shared<Tensor>(self->shape(), gi_strides, 0,
                                                std::move(grad_in_storage),
                                                self->backend(), self->dtype());
        self->accumulate_grad(grad_in);
      }
    };
  }
  return out;
}

std::shared_ptr<Tensor>
Tensor::embedding(const std::shared_ptr<Tensor> &weight) {
  if (shape_.size() != 2) {
    throw std::runtime_error("embedding indices expects a 2D tensor of shape "
                             "[batch_size, seq_len].");
  }
  if (weight->shape().size() != 2) {
    throw std::runtime_error("embedding weight expects a 2D tensor of shape "
                             "[vocab_size, embedding_dim].");
  }

  size_t N = shape_[0];
  size_t T = shape_[1];
  size_t D = weight->shape()[1];

  Shape out_shape = {N, T, D};
  auto out_storage =
      backend_->allocate(count_elements(out_shape) * dtype_size(weight->dtype()));
  out_storage->set_dtype(weight->dtype());
  Strides out_strides = compute_strides(out_shape);

  backend_->embedding(*weight->storage(), weight->offset(), weight->shape(),
                      weight->strides(), *storage_, offset_, shape_, strides_,
                      *out_storage, 0, out_shape, out_strides);

  auto out = std::make_shared<Tensor>(out_shape, out_strides, 0,
                                      std::move(out_storage), backend_, weight->dtype());
  out->requires_grad_ = weight->requires_grad();

  if (out->requires_grad_) {
    out->inputs_ = {shared_from_this(), weight};
    out->backward_ = [self = shared_from_this(), weight, out]() {
      if (weight->requires_grad()) {
        auto grad_w_storage = weight->backend()->allocate(
            count_elements(weight->shape()) * dtype_size(weight->dtype()));
        grad_w_storage->set_dtype(weight->dtype());
        Strides gw_strides = compute_strides(weight->shape());

        self->backend()->embedding_backward(
            *out->grad()->storage(), out->grad()->offset(),
            out->grad()->shape(), out->grad()->strides(), *self->storage(),
            self->offset(), self->shape(), self->strides(), *grad_w_storage, 0,
            weight->shape(), gw_strides);

        auto grad_w = std::make_shared<Tensor>(weight->shape(), gw_strides, 0,
                                               std::move(grad_w_storage),
                                               weight->backend(), weight->dtype());
        weight->accumulate_grad(grad_w);
      }
    };
  }
  return out;
}

static void topological_sort(const std::shared_ptr<Tensor> &node,
                             std::vector<std::shared_ptr<Tensor>> &sorted,
                             std::unordered_set<Tensor *> &visited) {
  if (visited.contains(node.get())) {
    return;
  }
  visited.insert(node.get());
  for (const auto &input : node->inputs()) {
    topological_sort(input, sorted, visited);
  }
  sorted.push_back(node);
}

std::shared_ptr<Tensor> Tensor::slice(size_t axis, size_t index) {
  Shape new_shape;
  Strides new_strides;
  new_shape.reserve(shape_.size() - 1);
  new_strides.reserve(strides_.size() - 1);
  for (size_t d = 0; d < shape_.size(); ++d) {
    if (d != axis) {
      new_shape.push_back(shape_[d]);
      new_strides.push_back(strides_[d]);
    }
  }

  size_t new_offset = offset_ + index * strides_[axis];

  auto out = std::make_shared<Tensor>(new_shape, new_strides, new_offset,
                                      storage_, backend_, dtype_);
  out->requires_grad_ = requires_grad_;

  if (out->requires_grad_) {
    out->inputs_ = {shared_from_this()};
    out->backward_ = [self = shared_from_this(), out, axis, index]() {
      if (self->requires_grad_) {
        auto gp_storage = self->backend()->allocate(
            count_elements(self->shape()) * dtype_size(self->dtype()));
        gp_storage->set_dtype(self->dtype());
        self->backend()->fill(*gp_storage, 0.0f);

        Strides gp_strides = compute_strides(self->shape());
        self->backend()->slice_backward(
            *out->grad()->storage(), out->grad()->offset(),
            out->grad()->shape(), out->grad()->strides(), *gp_storage, 0,
            self->shape(), gp_strides, axis, index);

        auto parent_grad =
            std::make_shared<Tensor>(self->shape(), gp_strides, 0,
                                     std::move(gp_storage), self->backend(), self->dtype());
        self->accumulate_grad(parent_grad);
      }
    };
  }

  return out;
}

void Tensor::backward() {
  std::vector<std::shared_ptr<Tensor>> sorted;
  std::unordered_set<Tensor *> visited;
  topological_sort(shared_from_this(), sorted, visited);

  if (!grad_) {
    grad_ =
        std::make_shared<Tensor>(Shape{}, std::vector<float>{1.0f}, backend_, dtype_);
  } else {
    grad_->fill(1.0f);
  }

  for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
    if ((*it)->backward_) {
      (*it)->backward_();
    }
  }
}

void Tensor::zero_grad() noexcept { grad_.reset(); }

void Tensor::accumulate_grad(const std::shared_ptr<Tensor> &incoming_grad) {
  if (!requires_grad_)
    return;
  if (!grad_) {
    grad_ = std::make_shared<Tensor>(shape_, backend_, dtype_);
    grad_->fill(0.0f);
  }
  backend_->add(*grad_->storage(), grad_->offset(), grad_->shape(),
                grad_->strides(), *incoming_grad->storage(),
                incoming_grad->offset(), incoming_grad->shape(),
                incoming_grad->strides(), *grad_->storage(), grad_->offset(),
                grad_->shape(), grad_->strides());
}

std::shared_ptr<Tensor>
Tensor::stack(const std::vector<std::shared_ptr<Tensor>> &tensors) {
  if (tensors.empty()) {
    throw std::runtime_error("stack: empty vector of tensors");
  }
  const auto &base_shape = tensors[0]->shape();
  Shape out_shape = {tensors.size()};
  out_shape.insert(out_shape.end(), base_shape.begin(), base_shape.end());

  size_t single_elements = count_elements(base_shape);
  DataType dt = tensors[0]->dtype();
  auto out_storage = tensors[0]->backend()->allocate(count_elements(out_shape) *
                                                     dtype_size(dt));
  out_storage->set_dtype(dt);

  for (size_t i = 0; i < tensors.size(); ++i) {
    std::vector<float> host_data = tensors[i]->to_host();
    if (dt == DataType::FP32) {
      tensors[0]->backend()->copy_to_device(host_data.data(), *out_storage,
                                            i * single_elements * sizeof(float),
                                            single_elements * sizeof(float));
    } else {
      std::vector<uint16_t> half_data(host_data.size());
      if (dt == DataType::FP16) {
        for (size_t j = 0; j < host_data.size(); ++j) {
          half_data[j] = float_to_fp16(host_data[j]);
        }
      } else {
        for (size_t j = 0; j < host_data.size(); ++j) {
          half_data[j] = float_to_bfloat16(host_data[j]);
        }
      }
      tensors[0]->backend()->copy_to_device(reinterpret_cast<const float *>(half_data.data()), *out_storage,
                                            i * single_elements * sizeof(uint16_t),
                                            single_elements * sizeof(uint16_t));
    }
  }

  Strides out_strides = compute_strides(out_shape);
  auto out = std::make_shared<Tensor>(
      out_shape, out_strides, 0, std::move(out_storage), tensors[0]->backend(), dt);

  bool req_grad = false;
  for (const auto &t : tensors) {
    if (t->requires_grad()) {
      req_grad = true;
    }
  }
  out->set_requires_grad(req_grad);

  if (out->requires_grad()) {
    out->inputs_ = tensors;
    out->backward_ = [tensors, out]() {
      for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i]->requires_grad()) {
          auto gs_tensor = out->grad()->slice(0, i);
          tensors[i]->accumulate_grad(gs_tensor);
        }
      }
    };
  }

  return out;
}

std::shared_ptr<Tensor> Tensor::rope(size_t S_past) {
  if (shape_.size() != 4) {
    throw std::runtime_error(
        "rope: input tensor must be 4D of shape [B, H, S, D]");
  }

  size_t B = shape_[0];
  size_t H = shape_[1];
  size_t S = shape_[2];
  size_t D = shape_[3];

  if (D % 2 != 0) {
    throw std::runtime_error("rope: head dimension D must be even");
  }

  std::vector<float> in_data = to_host();
  std::vector<float> out_data(in_data.size());

  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t s = 0; s < S; ++s) {
        float m = static_cast<float>(S_past + s);
        for (size_t j = 0; j < D / 2; ++j) {
          float theta = std::pow(10000.0f, -2.0f * static_cast<float>(j) /
                                               static_cast<float>(D));
          float omega = m * theta;
          float cos_val = std::cos(omega);
          float sin_val = std::sin(omega);

          size_t idx_even = b * (H * S * D) + h * (S * D) + s * D + 2 * j;
          size_t idx_odd = idx_even + 1;

          float x_even = in_data[idx_even];
          float x_odd = in_data[idx_odd];

          out_data[idx_even] = x_even * cos_val - x_odd * sin_val;
          out_data[idx_odd] = x_even * sin_val + x_odd * cos_val;
        }
      }
    }
  }

  auto out_storage = backend_->allocate(out_data.size() * sizeof(float));
  backend_->copy_host_to_device(out_data.data(), *out_storage,
                                out_data.size() * sizeof(float));

  auto out = std::make_shared<Tensor>(shape_, compute_strides(shape_), 0,
                                      std::move(out_storage), backend_);
  out->requires_grad_ = requires_grad_;

  if (out->requires_grad_) {
    out->inputs_ = {shared_from_this()};
    out->backward_ = [self = shared_from_this(), out, S_past]() {
      if (self->requires_grad_) {
        std::vector<float> grad_out = out->grad()->to_host();
        std::vector<float> grad_in_data(grad_out.size());

        size_t B = self->shape()[0];
        size_t H = self->shape()[1];
        size_t S = self->shape()[2];
        size_t D = self->shape()[3];

        for (size_t b = 0; b < B; ++b) {
          for (size_t h = 0; h < H; ++h) {
            for (size_t s = 0; s < S; ++s) {
              float m = static_cast<float>(S_past + s);
              for (size_t j = 0; j < D / 2; ++j) {
                float theta = std::pow(10000.0f, -2.0f * static_cast<float>(j) /
                                                     static_cast<float>(D));
                float omega = m * theta;
                float cos_val = std::cos(omega);
                float sin_val = std::sin(omega);

                size_t idx_even = b * (H * S * D) + h * (S * D) + s * D + 2 * j;
                size_t idx_odd = idx_even + 1;

                float g_even = grad_out[idx_even];
                float g_odd = grad_out[idx_odd];

                grad_in_data[idx_even] = g_even * cos_val + g_odd * sin_val;
                grad_in_data[idx_odd] = -g_even * sin_val + g_odd * cos_val;
              }
            }
          }
        }

        auto grad_in_storage =
            self->backend()->allocate(grad_in_data.size() * sizeof(float));
        self->backend()->copy_host_to_device(
            grad_in_data.data(), *grad_in_storage,
            grad_in_data.size() * sizeof(float));
        auto grad_in = std::make_shared<Tensor>(
            self->shape(), compute_strides(self->shape()), 0,
            std::move(grad_in_storage), self->backend());
        self->accumulate_grad(grad_in);
      }
    };
  }

  return out;
}

std::shared_ptr<Tensor> Tensor::to(DataType target_dtype) {
  if (dtype_ == target_dtype) {
    return shared_from_this();
  }

  std::vector<float> fp32_data = to_host();
  auto out = std::make_shared<Tensor>(shape_, backend_, target_dtype);
  out->copy_from_host(fp32_data);

  out->requires_grad_ = requires_grad_;
  if (requires_grad_) {
    out->inputs_ = {shared_from_this()};
    out->backward_ = [self = shared_from_this(), out]() {
      if (self->requires_grad_ && out->grad()) {
        auto grad_in = out->grad()->to(self->dtype());
        self->accumulate_grad(grad_in);
      }
    };
  }
  return out;
}
