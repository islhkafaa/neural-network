#include "nn/quantization.hpp"
#include "optim/grad_scaler.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

QuantizedDense::QuantizedDense(std::shared_ptr<Dense> float_dense,
                               const QuantizationConfig &config)
    : type_(config.type), group_size_(config.group_size) {
  if (!float_dense) {
    throw std::runtime_error(
        "QuantizedDense requires a valid float Dense layer");
  }

  auto params = float_dense->parameters();
  if (params.empty()) {
    throw std::runtime_error("Base Dense layer has no parameters");
  }

  auto float_weight = params[0];
  weight_shape_ = float_weight->shape();
  backend_ = float_weight->backend();

  if (params.size() > 1) {
    bias_ = params[1];
    bias_->set_requires_grad(false);
  } else {
    bias_ = nullptr;
  }

  size_t in_features = weight_shape_[0];
  size_t out_features = weight_shape_[1];
  size_t total = in_features * out_features;

  auto weight_data = float_weight->to_host();
  std::vector<float> scale_factors(out_features, 1.0f);

  float max_q = (type_ == QuantizationType::INT8) ? 127.0f : 7.0f;

  for (size_t c = 0; c < out_features; ++c) {
    float max_val = 0.0f;
    for (size_t r = 0; r < in_features; ++r) {
      float val = std::abs(weight_data[r * out_features + c]);
      if (val > max_val) {
        max_val = val;
      }
    }
    float s = max_val / max_q;
    scale_factors[c] = (s == 0.0f) ? 1.0f : s;
  }

  scales_ =
      std::make_shared<Tensor>(Shape{out_features}, scale_factors, backend_);
  scales_->set_requires_grad(false);

  if (type_ == QuantizationType::INT8) {
    packed_weights_.resize(total);
    for (size_t r = 0; r < in_features; ++r) {
      for (size_t c = 0; c < out_features; ++c) {
        size_t idx = r * out_features + c;
        float s = scale_factors[c];
        float q_f = std::round(weight_data[idx] / s);
        q_f = std::max(-127.0f, std::min(127.0f, q_f));
        packed_weights_[idx] = static_cast<uint8_t>(static_cast<int8_t>(q_f));
      }
    }
  } else {
    size_t packed_size = (total + 1) / 2;
    packed_weights_.assign(packed_size, 0);
    for (size_t r = 0; r < in_features; ++r) {
      for (size_t c = 0; c < out_features; ++c) {
        size_t idx = r * out_features + c;
        float s = scale_factors[c];
        float q_f = std::round(weight_data[idx] / s);
        q_f = std::max(-7.0f, std::min(7.0f, q_f));
        int8_t q = static_cast<int8_t>(q_f);

        uint8_t val = static_cast<uint8_t>(q) & 0x0F;
        if (idx % 2 == 0) {
          packed_weights_[idx / 2] = val;
        } else {
          packed_weights_[idx / 2] |= (val << 4);
        }
      }
    }
  }
}

std::shared_ptr<Tensor> QuantizedDense::dequantize_weight() const {
  size_t in_features = weight_shape_[0];
  size_t out_features = weight_shape_[1];
  size_t total = in_features * out_features;

  std::vector<float> dequantized(total);
  auto scales_data = scales_->to_host();

  if (type_ == QuantizationType::INT8) {
    for (size_t r = 0; r < in_features; ++r) {
      for (size_t c = 0; c < out_features; ++c) {
        size_t idx = r * out_features + c;
        int8_t Q = static_cast<int8_t>(packed_weights_[idx]);
        float s = scales_data[c];
        dequantized[idx] = static_cast<float>(Q) * s;
      }
    }
  } else {
    for (size_t r = 0; r < in_features; ++r) {
      for (size_t c = 0; c < out_features; ++c) {
        size_t idx = r * out_features + c;
        uint8_t byte = 0;
        if (idx % 2 == 0) {
          byte = packed_weights_[idx / 2] & 0x0F;
        } else {
          byte = (packed_weights_[idx / 2] >> 4) & 0x0F;
        }
        int8_t Q = (byte & 0x08) ? static_cast<int8_t>(byte | 0xF0)
                                 : static_cast<int8_t>(byte);
        float s = scales_data[c];
        dequantized[idx] = static_cast<float>(Q) * s;
      }
    }
  }

  return std::make_shared<Tensor>(weight_shape_, dequantized, backend_);
}

std::shared_ptr<Tensor>
QuantizedDense::forward(const std::shared_ptr<Tensor> &input) {
  auto dequant_w = dequantize_weight();

  if (AMPContext::is_enabled()) {
    auto dt = AMPContext::active_dtype();
    auto active_input = input->to(dt);
    auto active_w = dequant_w->to(dt);
    if (bias_) {
      auto active_bias = bias_->to(dt);
      return active_input->matmul(active_w)->add(active_bias);
    } else {
      return active_input->matmul(active_w);
    }
  }

  if (bias_) {
    return input->matmul(dequant_w)->add(bias_);
  }
  return input->matmul(dequant_w);
}

std::vector<std::shared_ptr<Tensor>> QuantizedDense::parameters() {
  if (bias_) {
    return {bias_};
  }
  return {};
}

void quantize_model(Layer *model, const QuantizationConfig &config) {
  if (!model)
    return;

  if (auto s = dynamic_cast<Sequential *>(model)) {
    for (auto &l : s->layers()) {
      if (auto d = std::dynamic_pointer_cast<Dense>(l)) {
        l = std::make_shared<QuantizedDense>(d, config);
      } else {
        quantize_model(l.get(), config);
      }
    }
  } else if (auto mha = dynamic_cast<MultiHeadAttention *>(model)) {
    mha->quantize(config);
  }
}
