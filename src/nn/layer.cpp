#include "nn/layer.hpp"
#include "backend/thread_pool.hpp"
#include "core/serialization.hpp"
#include "nn/initialization.hpp"
#include "nn/lora.hpp"
#include "nn/quantization.hpp"
#include "optim/grad_scaler.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

std::shared_ptr<Tensor> KVCache::dequantize_k(ExecutionBackend *backend,
                                              DataType dtype) const {
  if (!quantized) {
    return k;
  }
  size_t size = quantized_k.size();
  std::vector<float> host_data(size);
  size_t D = shape[3];
  size_t S_total = shape[2];
  size_t H_kv = shape[1];
  size_t B = shape[0];

  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H_kv; ++h) {
      for (size_t j = 0; j < S_total; ++j) {
        size_t scale_idx = b * (H_kv * S_total) + h * S_total + j;
        float scale = k_scales[scale_idx];
        for (size_t d = 0; d < D; ++d) {
          size_t idx = b * (H_kv * S_total * D) + h * (S_total * D) + j * D + d;
          host_data[idx] = static_cast<float>(quantized_k[idx]) * scale;
        }
      }
    }
  }
  return std::make_shared<Tensor>(shape, host_data, backend, dtype);
}

std::shared_ptr<Tensor> KVCache::dequantize_v(ExecutionBackend *backend,
                                              DataType dtype) const {
  if (!quantized) {
    return v;
  }
  size_t size = quantized_v.size();
  std::vector<float> host_data(size);
  size_t D = shape[3];
  size_t S_total = shape[2];
  size_t H_kv = shape[1];
  size_t B = shape[0];

  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H_kv; ++h) {
      for (size_t j = 0; j < S_total; ++j) {
        size_t scale_idx = b * (H_kv * S_total) + h * S_total + j;
        float scale = v_scales[scale_idx];
        for (size_t d = 0; d < D; ++d) {
          size_t idx = b * (H_kv * S_total * D) + h * (S_total * D) + j * D + d;
          host_data[idx] = static_cast<float>(quantized_v[idx]) * scale;
        }
      }
    }
  }
  return std::make_shared<Tensor>(shape, host_data, backend, dtype);
}

void KVCache::truncate(size_t len) {
  if (quantized) {
    if (quantized_k.empty()) {
      return;
    }
    size_t B = shape[0];
    size_t H_kv = shape[1];
    size_t S_past = shape[2];
    size_t D = shape[3];

    if (len >= S_past) {
      return;
    }

    std::vector<int8_t> concat_k(B * H_kv * len * D);
    std::vector<int8_t> concat_v(B * H_kv * len * D);
    std::vector<float> concat_k_scales(B * H_kv * len);
    std::vector<float> concat_v_scales(B * H_kv * len);

    for (size_t b = 0; b < B; ++b) {
      for (size_t h = 0; h < H_kv; ++h) {
        std::copy(quantized_k.begin() +
                      (b * H_kv * S_past * D + h * S_past * D),
                  quantized_k.begin() +
                      (b * H_kv * S_past * D + h * S_past * D + len * D),
                  concat_k.begin() + (b * H_kv * len * D + h * len * D));
        std::copy(quantized_v.begin() +
                      (b * H_kv * S_past * D + h * S_past * D),
                  quantized_v.begin() +
                      (b * H_kv * S_past * D + h * S_past * D + len * D),
                  concat_v.begin() + (b * H_kv * len * D + h * len * D));
        std::copy(k_scales.begin() + (b * H_kv * S_past + h * S_past),
                  k_scales.begin() + (b * H_kv * S_past + h * S_past + len),
                  concat_k_scales.begin() + (b * H_kv * len + h * len));
        std::copy(v_scales.begin() + (b * H_kv * S_past + h * S_past),
                  v_scales.begin() + (b * H_kv * S_past + h * S_past + len),
                  concat_v_scales.begin() + (b * H_kv * len + h * len));
      }
    }

    quantized_k = std::move(concat_k);
    quantized_v = std::move(concat_v);
    k_scales = std::move(concat_k_scales);
    v_scales = std::move(concat_v_scales);
    shape = Shape{B, H_kv, len, D};
  } else {
    if (!k) {
      return;
    }
    size_t B = k->shape()[0];
    size_t H_kv = k->shape()[1];
    size_t S_past = k->shape()[2];
    size_t D = k->shape()[3];

    if (len >= S_past) {
      return;
    }

    std::vector<float> past_k = k->to_host();
    std::vector<float> past_v = v->to_host();

    std::vector<float> concat_k(B * H_kv * len * D);
    std::vector<float> concat_v(B * H_kv * len * D);

    for (size_t b = 0; b < B; ++b) {
      for (size_t h = 0; h < H_kv; ++h) {
        std::copy(past_k.begin() + (b * H_kv * S_past * D + h * S_past * D),
                  past_k.begin() +
                      (b * H_kv * S_past * D + h * S_past * D + len * D),
                  concat_k.begin() + (b * H_kv * len * D + h * len * D));
        std::copy(past_v.begin() + (b * H_kv * S_past * D + h * S_past * D),
                  past_v.begin() +
                      (b * H_kv * S_past * D + h * S_past * D + len * D),
                  concat_v.begin() + (b * H_kv * len * D + h * len * D));
      }
    }

    k = std::make_shared<Tensor>(Shape{B, H_kv, len, D}, concat_k, k->backend(),
                                 k->dtype());
    v = std::make_shared<Tensor>(Shape{B, H_kv, len, D}, concat_v, v->backend(),
                                 v->dtype());
  }
}

Dense::Dense(size_t in_features, size_t out_features,
             ExecutionBackend *backend) {
  weight_ = std::make_shared<Tensor>(Shape{in_features, out_features}, backend);
  weight_->set_requires_grad(true);
  init::xavier_uniform(*weight_);

  bias_ = std::make_shared<Tensor>(Shape{1, out_features}, backend);
  bias_->set_requires_grad(true);
  init::constant(*bias_, 0.0f);
}

std::shared_ptr<Tensor> Dense::forward(const std::shared_ptr<Tensor> &input) {
  if (AMPContext::is_enabled()) {
    auto dt = AMPContext::active_dtype();
    auto active_input = input->to(dt);
    auto active_weight = weight_->to(dt);
    auto active_bias = bias_->to(dt);
    return active_input->matmul(active_weight)->add(active_bias);
  }
  return input->matmul(weight_)->add(bias_);
}

std::shared_ptr<Tensor> Dropout::forward(const std::shared_ptr<Tensor> &input) {
  if (!training_ || p_ <= 0.0f) {
    return input;
  }
  if (p_ >= 1.0f) {
    auto zero = std::make_shared<Tensor>(input->shape(), input->backend());
    zero->fill(0.0f);
    return zero;
  }

  auto mask = std::make_shared<Tensor>(input->shape(), input->backend());
  size_t size = count_elements(input->shape());
  std::vector<float> host_mask(size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution dis(1.0f - p_);
  float scale = 1.0f / (1.0f - p_);

  for (size_t i = 0; i < size; ++i) {
    host_mask[i] = dis(gen) ? scale : 0.0f;
  }
  mask->copy_from_host(host_mask);

  return input->mul(mask);
}

LayerNorm::LayerNorm(const Shape &normalized_shape, ExecutionBackend *backend,
                     float eps)
    : normalized_shape_(normalized_shape), eps_(eps) {
  gamma_ = std::make_shared<Tensor>(normalized_shape_, backend);
  gamma_->set_requires_grad(true);
  init::constant(*gamma_, 1.0f);

  beta_ = std::make_shared<Tensor>(normalized_shape_, backend);
  beta_->set_requires_grad(true);
  init::constant(*beta_, 0.0f);
}

std::shared_ptr<Tensor>
LayerNorm::forward(const std::shared_ptr<Tensor> &input) {
  if (AMPContext::is_enabled()) {
    auto dt = AMPContext::active_dtype();
    auto active_input = input->to(dt);
    auto active_gamma = gamma_->to(dt);
    auto active_beta = beta_->to(dt);

    std::vector<size_t> axes;
    size_t start_axis = active_input->shape().size() - normalized_shape_.size();
    for (size_t a = start_axis; a < active_input->shape().size(); ++a) {
      axes.push_back(a);
    }

    float N = static_cast<float>(count_elements(normalized_shape_));
    auto N_tensor = std::make_shared<Tensor>(Shape{}, std::vector<float>{N},
                                             active_input->backend(), dt);
    auto eps_tensor = std::make_shared<Tensor>(
        Shape{}, std::vector<float>{eps_}, active_input->backend(), dt);

    auto mean = active_input->sum(axes, true)->div(N_tensor);
    auto diff = active_input->sub(mean);
    auto sq_diff = diff->mul(diff);
    auto var = sq_diff->sum(axes, true)->div(N_tensor);
    auto stddev = var->add(eps_tensor)->sqrt();
    auto x_hat = diff->div(stddev);

    return x_hat->mul(active_gamma)->add(active_beta);
  }

  std::vector<size_t> axes;
  size_t start_axis = input->shape().size() - normalized_shape_.size();
  for (size_t a = start_axis; a < input->shape().size(); ++a) {
    axes.push_back(a);
  }

  float N = static_cast<float>(count_elements(normalized_shape_));
  auto N_tensor = std::make_shared<Tensor>(Shape{}, std::vector<float>{N},
                                           input->backend());
  auto eps_tensor = std::make_shared<Tensor>(Shape{}, std::vector<float>{eps_},
                                             input->backend());

  auto mean = input->sum(axes, true)->div(N_tensor);
  auto diff = input->sub(mean);
  auto sq_diff = diff->mul(diff);
  auto var = sq_diff->sum(axes, true)->div(N_tensor);
  auto stddev = var->add(eps_tensor)->sqrt();
  auto x_hat = diff->div(stddev);

  return x_hat->mul(gamma_)->add(beta_);
}

Embedding::Embedding(size_t num_embeddings, size_t embedding_dim,
                     ExecutionBackend *backend) {
  weight_ =
      std::make_shared<Tensor>(Shape{num_embeddings, embedding_dim}, backend);
  weight_->set_requires_grad(true);
  init::normal(*weight_, 0.0f, 1.0f);
}

std::shared_ptr<Tensor>
Embedding::forward(const std::shared_ptr<Tensor> &input) {
  return input->embedding(weight_);
}

Conv2D::Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size,
               size_t stride, size_t padding, ExecutionBackend *backend)
    : stride_(stride), padding_(padding) {
  weight_ = std::make_shared<Tensor>(
      Shape{out_channels, in_channels, kernel_size, kernel_size}, backend);
  weight_->set_requires_grad(true);
  init::kaiming_uniform(*weight_);

  bias_ = std::make_shared<Tensor>(Shape{1, out_channels, 1, 1}, backend);
  bias_->set_requires_grad(true);
  init::constant(*bias_, 0.0f);
}

std::shared_ptr<Tensor> Conv2D::forward(const std::shared_ptr<Tensor> &input) {
  return input->conv2d(weight_, bias_, padding_, stride_);
}

std::shared_ptr<Tensor>
MaxPooling2D::forward(const std::shared_ptr<Tensor> &input) {
  return input->maxpool2d(pool_size_, pool_size_, stride_);
}

RNN::RNN(size_t input_size, size_t hidden_size, ExecutionBackend *backend)
    : hidden_size_(hidden_size) {
  W_ih_ = std::make_shared<Tensor>(Shape{input_size, hidden_size}, backend);
  W_hh_ = std::make_shared<Tensor>(Shape{hidden_size, hidden_size}, backend);
  b_ih_ = std::make_shared<Tensor>(Shape{1, hidden_size}, backend);
  b_hh_ = std::make_shared<Tensor>(Shape{1, hidden_size}, backend);

  W_ih_->set_requires_grad(true);
  W_hh_->set_requires_grad(true);
  b_ih_->set_requires_grad(true);
  b_hh_->set_requires_grad(true);

  init::xavier_uniform(*W_ih_);
  init::xavier_uniform(*W_hh_);
  init::constant(*b_ih_, 0.0f);
  init::constant(*b_hh_, 0.0f);
}

std::shared_ptr<Tensor> RNN::forward(const std::shared_ptr<Tensor> &input) {
  return forward(input, nullptr);
}

std::shared_ptr<Tensor> RNN::forward(const std::shared_ptr<Tensor> &input,
                                     const std::shared_ptr<Tensor> &h0) {
  size_t seq_len = input->shape()[0];
  size_t batch_size = input->shape()[1];
  auto backend = input->backend();

  std::shared_ptr<Tensor> h_prev = h0;
  if (!h_prev) {
    h_prev = std::make_shared<Tensor>(Shape{batch_size, hidden_size_}, backend);
    h_prev->fill(0.0f);
  }

  std::vector<std::shared_ptr<Tensor>> h_outputs;
  h_outputs.reserve(seq_len);

  for (size_t t = 0; t < seq_len; ++t) {
    auto xt = input->slice(0, t);
    auto xt_Wih = xt->matmul(W_ih_)->add(b_ih_);
    auto h_prev_Whh = h_prev->matmul(W_hh_)->add(b_hh_);
    auto h_t = xt_Wih->add(h_prev_Whh)->tanh();
    h_outputs.push_back(h_t);
    h_prev = h_t;
  }

  return Tensor::stack(h_outputs);
}

LSTM::LSTM(size_t input_size, size_t hidden_size, ExecutionBackend *backend)
    : hidden_size_(hidden_size) {
  auto make_gate = [&](std::shared_ptr<Tensor> &W_ih,
                       std::shared_ptr<Tensor> &W_hh,
                       std::shared_ptr<Tensor> &b_ih,
                       std::shared_ptr<Tensor> &b_hh) {
    W_ih = std::make_shared<Tensor>(Shape{input_size, hidden_size}, backend);
    W_hh = std::make_shared<Tensor>(Shape{hidden_size, hidden_size}, backend);
    b_ih = std::make_shared<Tensor>(Shape{1, hidden_size}, backend);
    b_hh = std::make_shared<Tensor>(Shape{1, hidden_size}, backend);

    W_ih->set_requires_grad(true);
    W_hh->set_requires_grad(true);
    b_ih->set_requires_grad(true);
    b_hh->set_requires_grad(true);

    init::xavier_uniform(*W_ih);
    init::xavier_uniform(*W_hh);
    init::constant(*b_ih, 0.0f);
    init::constant(*b_hh, 0.0f);
  };

  make_gate(W_ii_, W_hi_, b_ii_, b_hi_);
  make_gate(W_if_, W_hf_, b_if_, b_hf_);
  make_gate(W_ig_, W_hg_, b_ig_, b_hg_);
  make_gate(W_io_, W_ho_, b_io_, b_ho_);
}

std::shared_ptr<Tensor> LSTM::forward(const std::shared_ptr<Tensor> &input) {
  auto hx = forward(input, {nullptr, nullptr});
  return hx.first;
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> LSTM::forward(
    const std::shared_ptr<Tensor> &input,
    const std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> &hx) {
  size_t seq_len = input->shape()[0];
  size_t batch_size = input->shape()[1];
  auto backend = input->backend();

  std::shared_ptr<Tensor> h_prev = hx.first;
  std::shared_ptr<Tensor> c_prev = hx.second;

  if (!h_prev) {
    h_prev = std::make_shared<Tensor>(Shape{batch_size, hidden_size_}, backend);
    h_prev->fill(0.0f);
  }
  if (!c_prev) {
    c_prev = std::make_shared<Tensor>(Shape{batch_size, hidden_size_}, backend);
    c_prev->fill(0.0f);
  }

  std::vector<std::shared_ptr<Tensor>> h_outputs;
  h_outputs.reserve(seq_len);

  for (size_t t = 0; t < seq_len; ++t) {
    auto xt = input->slice(0, t);

    auto i_t = xt->matmul(W_ii_)
                   ->add(b_ii_)
                   ->add(h_prev->matmul(W_hi_)->add(b_hi_))
                   ->sigmoid();

    auto f_t = xt->matmul(W_if_)
                   ->add(b_if_)
                   ->add(h_prev->matmul(W_hf_)->add(b_hf_))
                   ->sigmoid();

    auto g_t = xt->matmul(W_ig_)
                   ->add(b_ig_)
                   ->add(h_prev->matmul(W_hg_)->add(b_hg_))
                   ->tanh();

    auto o_t = xt->matmul(W_io_)
                   ->add(b_io_)
                   ->add(h_prev->matmul(W_ho_)->add(b_ho_))
                   ->sigmoid();

    auto c_t = f_t->mul(c_prev)->add(i_t->mul(g_t));

    auto h_t = o_t->mul(c_t->tanh());

    h_outputs.push_back(h_t);
    h_prev = h_t;
    c_prev = c_t;
  }

  return {Tensor::stack(h_outputs), h_prev};
}

std::shared_ptr<Tensor>
Sequential::forward(const std::shared_ptr<Tensor> &input) {
  auto current = input;
  for (auto &layer : layers_) {
    current = layer->forward(current);
  }
  return current;
}

void Sequential::set_training(bool training) {
  Layer::set_training(training);
  for (auto &layer : layers_) {
    layer->set_training(training);
  }
}

std::vector<std::shared_ptr<Tensor>> Sequential::parameters() {
  std::vector<std::shared_ptr<Tensor>> params;
  for (auto &layer : layers_) {
    auto p = layer->parameters();
    params.insert(params.end(), p.begin(), p.end());
  }
  return params;
}

void Layer::save_weights(const std::string &filepath) {
  serialization::save_parameters(this->parameters(), filepath);
}

void Layer::load_weights(const std::string &filepath) {
  serialization::load_parameters(this->parameters(), filepath);
}

MultiHeadAttention::MultiHeadAttention(size_t embed_dim, size_t num_heads,
                                       bool causal, ExecutionBackend *backend,
                                       size_t num_kv_heads)
    : num_heads_(num_heads),
      num_kv_heads_(num_kv_heads == 0 ? num_heads : num_kv_heads),
      head_dim_(embed_dim / num_heads), causal_(causal) {
  if (embed_dim % num_heads != 0) {
    throw std::runtime_error("embed_dim must be divisible by num_heads");
  }
  if (num_heads_ % num_kv_heads_ != 0) {
    throw std::runtime_error("num_heads must be divisible by num_kv_heads");
  }
  q_proj_ = std::make_shared<Dense>(embed_dim, embed_dim, backend);
  k_proj_ =
      std::make_shared<Dense>(embed_dim, num_kv_heads_ * head_dim_, backend);
  v_proj_ =
      std::make_shared<Dense>(embed_dim, num_kv_heads_ * head_dim_, backend);
  out_proj_ = std::make_shared<Dense>(embed_dim, embed_dim, backend);
}

std::vector<std::shared_ptr<Tensor>> MultiHeadAttention::parameters() {
  std::vector<std::shared_ptr<Tensor>> params;
  auto q_p = q_proj_->parameters();
  auto k_p = k_proj_->parameters();
  auto v_p = v_proj_->parameters();
  auto o_p = out_proj_->parameters();
  params.insert(params.end(), q_p.begin(), q_p.end());
  params.insert(params.end(), k_p.begin(), k_p.end());
  params.insert(params.end(), v_p.begin(), v_p.end());
  params.insert(params.end(), o_p.begin(), o_p.end());
  return params;
}

void MultiHeadAttention::apply_lora(const LoRAConfig &cfg) {
  if (cfg.target_modules.count("q_proj")) {
    if (auto d = std::dynamic_pointer_cast<Dense>(q_proj_)) {
      q_proj_ = std::make_shared<LoRALinear>(d, cfg);
    }
  }
  if (cfg.target_modules.count("k_proj")) {
    if (auto d = std::dynamic_pointer_cast<Dense>(k_proj_)) {
      k_proj_ = std::make_shared<LoRALinear>(d, cfg);
    }
  }
  if (cfg.target_modules.count("v_proj")) {
    if (auto d = std::dynamic_pointer_cast<Dense>(v_proj_)) {
      v_proj_ = std::make_shared<LoRALinear>(d, cfg);
    }
  }
  if (cfg.target_modules.count("out_proj")) {
    if (auto d = std::dynamic_pointer_cast<Dense>(out_proj_)) {
      out_proj_ = std::make_shared<LoRALinear>(d, cfg);
    }
  }
}

std::vector<std::shared_ptr<LoRALinear>>
MultiHeadAttention::lora_modules() const {
  std::vector<std::shared_ptr<LoRALinear>> modules;
  if (auto l = std::dynamic_pointer_cast<LoRALinear>(q_proj_)) {
    modules.push_back(l);
  }
  if (auto l = std::dynamic_pointer_cast<LoRALinear>(k_proj_)) {
    modules.push_back(l);
  }
  if (auto l = std::dynamic_pointer_cast<LoRALinear>(v_proj_)) {
    modules.push_back(l);
  }
  if (auto l = std::dynamic_pointer_cast<LoRALinear>(out_proj_)) {
    modules.push_back(l);
  }
  return modules;
}

void MultiHeadAttention::quantize(const QuantizationConfig &cfg) {
  if (auto d = std::dynamic_pointer_cast<Dense>(q_proj_)) {
    q_proj_ = std::make_shared<QuantizedDense>(d, cfg);
  }
  if (auto d = std::dynamic_pointer_cast<Dense>(k_proj_)) {
    k_proj_ = std::make_shared<QuantizedDense>(d, cfg);
  }
  if (auto d = std::dynamic_pointer_cast<Dense>(v_proj_)) {
    v_proj_ = std::make_shared<QuantizedDense>(d, cfg);
  }
  if (auto d = std::dynamic_pointer_cast<Dense>(out_proj_)) {
    out_proj_ = std::make_shared<QuantizedDense>(d, cfg);
  }
}

std::shared_ptr<Tensor>
MultiHeadAttention::forward(const std::shared_ptr<Tensor> &input) {
  return forward(input, nullptr);
}

static std::shared_ptr<Tensor> expand_heads(const std::shared_ptr<Tensor> &src,
                                            size_t num_heads,
                                            size_t num_kv_heads) {
  if (num_heads == num_kv_heads) {
    return src;
  }
  size_t B = src->shape()[0];
  size_t S = src->shape()[2];
  size_t D = src->shape()[3];
  size_t group_size = num_heads / num_kv_heads;

  std::vector<float> src_data = src->to_host();
  std::vector<float> dst_data(B * num_heads * S * D);

  for (size_t b = 0; b < B; ++b) {
    for (size_t h_q = 0; h_q < num_heads; ++h_q) {
      size_t h_kv = h_q / group_size;
      std::copy(src_data.begin() + (b * num_kv_heads * S * D + h_kv * S * D),
                src_data.begin() +
                    (b * num_kv_heads * S * D + h_kv * S * D + S * D),
                dst_data.begin() + (b * num_heads * S * D + h_q * S * D));
    }
  }

  return std::make_shared<Tensor>(Shape{B, num_heads, S, D}, dst_data,
                                  src->backend());
}

std::shared_ptr<Tensor>
MultiHeadAttention::forward(const std::shared_ptr<Tensor> &input,
                            KVCache *cache) {
  if (input->shape().size() != 3) {
    throw std::runtime_error(
        "MultiHeadAttention input must be of shape (B, S, E)");
  }
  size_t B = input->shape()[0];
  size_t S = input->shape()[1];
  size_t E = input->shape()[2];

  auto x_flat = input->reshape(Shape{B * S, E});

  auto q_flat = q_proj_->forward(x_flat);
  auto k_flat = k_proj_->forward(x_flat);
  auto v_flat = v_proj_->forward(x_flat);

  auto q = q_flat->reshape(Shape{B, S, num_heads_, head_dim_});
  auto k = k_flat->reshape(Shape{B, S, num_kv_heads_, head_dim_});
  auto v = v_flat->reshape(Shape{B, S, num_kv_heads_, head_dim_});

  auto q_t = q->transpose(1, 2);
  auto k_t = k->transpose(1, 2);
  auto v_t = v->transpose(1, 2);

  size_t S_past = 0;
  if (cache) {
    if (cache->quantized) {
      if (!cache->quantized_k.empty()) {
        S_past = cache->shape[2];
      }
    } else if (cache->k) {
      S_past = cache->k->shape()[2];
    }
  }

  q_t = q_t->rope(S_past);
  k_t = k_t->rope(S_past);

  std::shared_ptr<Tensor> active_k;
  std::shared_ptr<Tensor> active_v;

  auto quantize_kv = [](const std::vector<float> &src, size_t B, size_t H_kv,
                        size_t S, size_t D, std::vector<int8_t> &dst_q,
                        std::vector<float> &dst_scales) {
    dst_q.resize(B * H_kv * S * D);
    dst_scales.resize(B * H_kv * S);
    for (size_t b = 0; b < B; ++b) {
      for (size_t h = 0; h < H_kv; ++h) {
        for (size_t s = 0; s < S; ++s) {
          size_t base_idx = b * (H_kv * S * D) + h * (S * D) + s * D;
          float max_val = 0.0f;
          for (size_t d = 0; d < D; ++d) {
            float val = std::abs(src[base_idx + d]);
            if (val > max_val) {
              max_val = val;
            }
          }
          float scale = max_val / 127.0f;
          if (scale == 0.0f) {
            scale = 1.0f;
          }
          dst_scales[b * (H_kv * S) + h * S + s] = scale;
          float inv_scale = 1.0f / scale;
          for (size_t d = 0; d < D; ++d) {
            float q_val = std::round(src[base_idx + d] * inv_scale);
            q_val = std::max(-127.0f, std::min(127.0f, q_val));
            dst_q[base_idx + d] = static_cast<int8_t>(q_val);
          }
        }
      }
    }
  };

  if (cache) {
    if (cache->quantized) {
      size_t D = head_dim_;
      if (cache->quantized_k.empty()) {
        std::vector<float> host_k = k_t->to_host();
        std::vector<float> host_v = v_t->to_host();
        quantize_kv(host_k, B, num_kv_heads_, S, D, cache->quantized_k,
                    cache->k_scales);
        quantize_kv(host_v, B, num_kv_heads_, S, D, cache->quantized_v,
                    cache->v_scales);
        cache->shape = Shape{B, num_kv_heads_, S, D};
      } else {
        size_t S_past = cache->shape[2];
        size_t S_total = S_past + S;

        std::vector<float> host_new_k = k_t->to_host();
        std::vector<float> host_new_v = v_t->to_host();

        std::vector<int8_t> new_q_k;
        std::vector<float> new_k_scales;
        quantize_kv(host_new_k, B, num_kv_heads_, S, D, new_q_k, new_k_scales);

        std::vector<int8_t> new_q_v;
        std::vector<float> new_v_scales;
        quantize_kv(host_new_v, B, num_kv_heads_, S, D, new_q_v, new_v_scales);

        std::vector<int8_t> concat_k(B * num_kv_heads_ * S_total * D);
        std::vector<int8_t> concat_v(B * num_kv_heads_ * S_total * D);
        std::vector<float> concat_k_scales(B * num_kv_heads_ * S_total);
        std::vector<float> concat_v_scales(B * num_kv_heads_ * S_total);

        for (size_t b = 0; b < B; ++b) {
          for (size_t h = 0; h < num_kv_heads_; ++h) {
            std::copy(cache->quantized_k.begin() +
                          (b * num_kv_heads_ * S_past * D + h * S_past * D),
                      cache->quantized_k.begin() +
                          (b * num_kv_heads_ * S_past * D + h * S_past * D +
                           S_past * D),
                      concat_k.begin() +
                          (b * num_kv_heads_ * S_total * D + h * S_total * D));
            std::copy(cache->quantized_v.begin() +
                          (b * num_kv_heads_ * S_past * D + h * S_past * D),
                      cache->quantized_v.begin() +
                          (b * num_kv_heads_ * S_past * D + h * S_past * D +
                           S_past * D),
                      concat_v.begin() +
                          (b * num_kv_heads_ * S_total * D + h * S_total * D));

            std::copy(new_q_k.begin() + (b * num_kv_heads_ * S * D + h * S * D),
                      new_q_k.begin() +
                          (b * num_kv_heads_ * S * D + h * S * D + S * D),
                      concat_k.begin() + (b * num_kv_heads_ * S_total * D +
                                          h * S_total * D + S_past * D));
            std::copy(new_q_v.begin() + (b * num_kv_heads_ * S * D + h * S * D),
                      new_q_v.begin() +
                          (b * num_kv_heads_ * S * D + h * S * D + S * D),
                      concat_v.begin() + (b * num_kv_heads_ * S_total * D +
                                          h * S_total * D + S_past * D));

            std::copy(cache->k_scales.begin() +
                          (b * num_kv_heads_ * S_past + h * S_past),
                      cache->k_scales.begin() +
                          (b * num_kv_heads_ * S_past + h * S_past + S_past),
                      concat_k_scales.begin() +
                          (b * num_kv_heads_ * S_total + h * S_total));
            std::copy(cache->v_scales.begin() +
                          (b * num_kv_heads_ * S_past + h * S_past),
                      cache->v_scales.begin() +
                          (b * num_kv_heads_ * S_past + h * S_past + S_past),
                      concat_v_scales.begin() +
                          (b * num_kv_heads_ * S_total + h * S_total));

            std::copy(new_k_scales.begin() + (b * num_kv_heads_ * S + h * S),
                      new_k_scales.begin() +
                          (b * num_kv_heads_ * S + h * S + S),
                      concat_k_scales.begin() +
                          (b * num_kv_heads_ * S_total + h * S_total + S_past));
            std::copy(new_v_scales.begin() + (b * num_kv_heads_ * S + h * S),
                      new_v_scales.begin() +
                          (b * num_kv_heads_ * S + h * S + S),
                      concat_v_scales.begin() +
                          (b * num_kv_heads_ * S_total + h * S_total + S_past));
          }
        }

        cache->quantized_k = std::move(concat_k);
        cache->quantized_v = std::move(concat_v);
        cache->k_scales = std::move(concat_k_scales);
        cache->v_scales = std::move(concat_v_scales);
        cache->shape = Shape{B, num_kv_heads_, S_total, D};
      }
    } else {
      if (!cache->k) {
        cache->k = k_t;
        cache->v = v_t;
      } else {
        size_t S_past = cache->k->shape()[2];
        size_t S_total = S_past + S;
        size_t D = head_dim_;

        std::vector<float> past_k = cache->k->to_host();
        std::vector<float> new_k = k_t->to_host();
        std::vector<float> concat_k(B * num_kv_heads_ * S_total * D);

        std::vector<float> past_v = cache->v->to_host();
        std::vector<float> new_v = v_t->to_host();
        std::vector<float> concat_v(B * num_kv_heads_ * S_total * D);

        for (size_t b = 0; b < B; ++b) {
          for (size_t h = 0; h < num_kv_heads_; ++h) {
            std::copy(past_k.begin() +
                          (b * num_kv_heads_ * S_past * D + h * S_past * D),
                      past_k.begin() + (b * num_kv_heads_ * S_past * D +
                                        h * S_past * D + S_past * D),
                      concat_k.begin() +
                          (b * num_kv_heads_ * S_total * D + h * S_total * D));
            std::copy(new_k.begin() + (b * num_kv_heads_ * S * D + h * S * D),
                      new_k.begin() +
                          (b * num_kv_heads_ * S * D + h * S * D + S * D),
                      concat_k.begin() + (b * num_kv_heads_ * S_total * D +
                                          h * S_total * D + S_past * D));

            std::copy(past_v.begin() +
                          (b * num_kv_heads_ * S_past * D + h * S_past * D),
                      past_v.begin() + (b * num_kv_heads_ * S_past * D +
                                        h * S_past * D + S_past * D),
                      concat_v.begin() +
                          (b * num_kv_heads_ * S_total * D + h * S_total * D));
            std::copy(new_v.begin() + (b * num_kv_heads_ * S * D + h * S * D),
                      new_v.begin() +
                          (b * num_kv_heads_ * S * D + h * S * D + S * D),
                      concat_v.begin() + (b * num_kv_heads_ * S_total * D +
                                          h * S_total * D + S_past * D));
          }
        }

        cache->k = std::make_shared<Tensor>(Shape{B, num_kv_heads_, S_total, D},
                                            concat_k, input->backend());
        cache->v = std::make_shared<Tensor>(Shape{B, num_kv_heads_, S_total, D},
                                            concat_v, input->backend());
      }
    }
  }

  bool use_fused = !q_t->requires_grad();
  if (cache) {
    if (cache->quantized) {
    } else {
      if (cache->k->requires_grad() || cache->v->requires_grad()) {
        use_fused = false;
      }
    }
  } else {
    if (k_t->requires_grad() || v_t->requires_grad()) {
      use_fused = false;
    }
  }

  if (cache) {
    if (cache->quantized) {
      if (!use_fused) {
        active_k = cache->dequantize_k(input->backend(), input->dtype());
        active_v = cache->dequantize_v(input->backend(), input->dtype());
      }
    } else {
      active_k = cache->k;
      active_v = cache->v;
    }
  } else {
    active_k = k_t;
    active_v = v_t;
  }

  std::shared_ptr<Tensor> context_flat;

  if (use_fused) {
    size_t S_total =
        (cache && cache->quantized) ? cache->shape[2] : active_k->shape()[2];
    size_t D = head_dim_;
    size_t H = num_heads_;
    size_t H_kv = num_kv_heads_;
    size_t group_size = H / H_kv;

    std::vector<float> q_data = q_t->to_host();
    std::vector<float> k_data;
    std::vector<float> v_data;
    if (!cache || !cache->quantized) {
      k_data = active_k->to_host();
      v_data = active_v->to_host();
    }
    std::vector<float> out_data(B * S * H * D, 0.0f);

    ThreadPool::instance().parallel_for(
        0, B * H, [&](size_t start, size_t end) {
          for (size_t bh = start; bh < end; ++bh) {
            size_t b = bh / H;
            size_t h = bh % H;
            size_t h_kv = h / group_size;

            std::vector<float> row_scores(S_total);
            float scale = std::sqrt(static_cast<float>(D));

            for (size_t i = 0; i < S; ++i) {
              float max_score = -std::numeric_limits<float>::infinity();

              for (size_t j = 0; j < S_total; ++j) {
                if (causal_ && j > S_past + i) {
                  row_scores[j] =
                      (q_t->dtype() == DataType::FP16) ? -65000.0f : -1e9f;
                  continue;
                }

                float dot = 0.0f;
                size_t q_base = b * (H * S * D) + h * (S * D) + i * D;
                if (cache && cache->quantized) {
                  size_t k_base =
                      b * (H_kv * S_total * D) + h_kv * (S_total * D) + j * D;
                  size_t scale_idx = b * (H_kv * S_total) + h_kv * S_total + j;
                  float k_scale = cache->k_scales[scale_idx];
                  for (size_t d = 0; d < D; ++d) {
                    dot += q_data[q_base + d] *
                           (static_cast<float>(cache->quantized_k[k_base + d]) *
                            k_scale);
                  }
                } else {
                  size_t k_base =
                      b * (H_kv * S_total * D) + h_kv * (S_total * D) + j * D;
                  for (size_t d = 0; d < D; ++d) {
                    dot += q_data[q_base + d] * k_data[k_base + d];
                  }
                }
                float score = dot / scale;
                row_scores[j] = score;
                if (score > max_score) {
                  max_score = score;
                }
              }

              float sum_exp = 0.0f;
              for (size_t j = 0; j < S_total; ++j) {
                if (causal_ && j > S_past + i) {
                  row_scores[j] = 0.0f;
                  continue;
                }
                float val = std::exp(row_scores[j] - max_score);
                row_scores[j] = val;
                sum_exp += val;
              }

              if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
                for (size_t j = 0; j < S_total; ++j) {
                  row_scores[j] *= inv_sum;
                }
              }

              size_t out_base = b * (S * H * D) + i * (H * D) + h * D;
              for (size_t d = 0; d < D; ++d) {
                float sum_val = 0.0f;
                for (size_t j = 0; j < S_total; ++j) {
                  if (row_scores[j] == 0.0f)
                    continue;
                  if (cache && cache->quantized) {
                    size_t v_idx = b * (H_kv * S_total * D) +
                                   h_kv * (S_total * D) + j * D + d;
                    size_t scale_idx =
                        b * (H_kv * S_total) + h_kv * S_total + j;
                    float v_scale = cache->v_scales[scale_idx];
                    sum_val += row_scores[j] *
                               (static_cast<float>(cache->quantized_v[v_idx]) *
                                v_scale);
                  } else {
                    size_t v_idx = b * (H_kv * S_total * D) +
                                   h_kv * (S_total * D) + j * D + d;
                    sum_val += row_scores[j] * v_data[v_idx];
                  }
                }
                out_data[out_base + d] = sum_val;
              }
            }
          }
        });

    auto context_t = std::make_shared<Tensor>(Shape{B, S, H, D}, out_data,
                                              input->backend(), input->dtype());
    context_flat = context_t->reshape(Shape{B * S, E});
  } else {
    auto expanded_k = expand_heads(active_k, num_heads_, num_kv_heads_);
    auto expanded_v = expand_heads(active_v, num_heads_, num_kv_heads_);

    auto k_trans = expanded_k->transpose(2, 3);
    auto scores = q_t->bmm(k_trans);

    float scale = std::sqrt(static_cast<float>(head_dim_));
    auto scaled_scores = scores / scale;

    if (causal_) {
      size_t S_total = expanded_k->shape()[2];
      size_t S_past = S_total - S;
      auto host_data = scaled_scores->to_host();
      size_t H = num_heads_;
      for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
          for (size_t i = 0; i < S; ++i) {
            for (size_t j = S_past + i + 1; j < S_total; ++j) {
              size_t idx =
                  b * (H * S * S_total) + h * (S * S_total) + i * S_total + j;
              float mask_val = (scaled_scores->dtype() == DataType::FP16)
                                   ? -65000.0f
                                   : -1e9f;
              host_data[idx] = mask_val;
            }
          }
        }
      }
      scaled_scores->copy_from_host(host_data);
    }

    auto attn = scaled_scores->softmax(3);

    auto context = attn->bmm(expanded_v);
    auto context_t = context->transpose(1, 2);
    context_flat = context_t->reshape(Shape{B * S, E});
  }

  auto out_flat = out_proj_->forward(context_flat);
  return out_flat->reshape(Shape{B, S, E});
}

TransformerEncoderLayer::TransformerEncoderLayer(size_t embed_dim,
                                                 size_t num_heads,
                                                 size_t dim_feedforward,
                                                 float dropout_p,
                                                 ExecutionBackend *backend) {
  self_attn_ =
      std::make_shared<MultiHeadAttention>(embed_dim, num_heads, backend);
  linear1_ = std::make_shared<Dense>(embed_dim, dim_feedforward, backend);
  linear2_ = std::make_shared<Dense>(dim_feedforward, embed_dim, backend);
  norm1_ = std::make_shared<LayerNorm>(Shape{embed_dim}, backend);
  norm2_ = std::make_shared<LayerNorm>(Shape{embed_dim}, backend);
  dropout1_ = std::make_shared<Dropout>(dropout_p);
  dropout2_ = std::make_shared<Dropout>(dropout_p);
}

void TransformerEncoderLayer::set_training(bool training) {
  Layer::set_training(training);
  self_attn_->set_training(training);
  linear1_->set_training(training);
  linear2_->set_training(training);
  norm1_->set_training(training);
  norm2_->set_training(training);
  dropout1_->set_training(training);
  dropout2_->set_training(training);
}

std::vector<std::shared_ptr<Tensor>> TransformerEncoderLayer::parameters() {
  std::vector<std::shared_ptr<Tensor>> params;
  auto sa_p = self_attn_->parameters();
  auto l1_p = linear1_->parameters();
  auto l2_p = linear2_->parameters();
  auto n1_p = norm1_->parameters();
  auto n2_p = norm2_->parameters();
  params.insert(params.end(), sa_p.begin(), sa_p.end());
  params.insert(params.end(), l1_p.begin(), l1_p.end());
  params.insert(params.end(), l2_p.begin(), l2_p.end());
  params.insert(params.end(), n1_p.begin(), n1_p.end());
  params.insert(params.end(), n2_p.begin(), n2_p.end());
  return params;
}

std::shared_ptr<Tensor>
TransformerEncoderLayer::forward(const std::shared_ptr<Tensor> &input) {
  auto attn_out = self_attn_->forward(input);
  auto dropped_attn = dropout1_->forward(attn_out);
  auto x1 = norm1_->forward(input->add(dropped_attn));

  size_t B = x1->shape()[0];
  size_t S = x1->shape()[1];
  size_t E = x1->shape()[2];

  auto x1_flat = x1->reshape(Shape{B * S, E});
  auto ffn_flat = linear2_->forward(linear1_->forward(x1_flat)->relu());
  auto ffn_out = ffn_flat->reshape(Shape{B, S, E});

  auto dropped_ffn = dropout2_->forward(ffn_out);
  return norm2_->forward(x1->add(dropped_ffn));
}

TransformerDecoderLayer::TransformerDecoderLayer(size_t embed_dim,
                                                 size_t num_heads,
                                                 size_t dim_feedforward,
                                                 float dropout_p,
                                                 ExecutionBackend *backend) {
  self_attn_ =
      std::make_shared<MultiHeadAttention>(embed_dim, num_heads, true, backend);
  ffn_ = std::make_shared<SwiGLU>(embed_dim, dim_feedforward, backend);
  norm1_ = std::make_shared<LayerNorm>(Shape{embed_dim}, backend);
  norm2_ = std::make_shared<LayerNorm>(Shape{embed_dim}, backend);
  dropout1_ = std::make_shared<Dropout>(dropout_p);
  dropout2_ = std::make_shared<Dropout>(dropout_p);
}

void TransformerDecoderLayer::set_training(bool training) {
  Layer::set_training(training);
  self_attn_->set_training(training);
  ffn_->set_training(training);
  norm1_->set_training(training);
  norm2_->set_training(training);
  dropout1_->set_training(training);
  dropout2_->set_training(training);
}

std::vector<std::shared_ptr<Tensor>> TransformerDecoderLayer::parameters() {
  std::vector<std::shared_ptr<Tensor>> params;
  auto sa_p = self_attn_->parameters();
  auto ffn_p = ffn_->parameters();
  auto n1_p = norm1_->parameters();
  auto n2_p = norm2_->parameters();
  params.insert(params.end(), sa_p.begin(), sa_p.end());
  params.insert(params.end(), ffn_p.begin(), ffn_p.end());
  params.insert(params.end(), n1_p.begin(), n1_p.end());
  params.insert(params.end(), n2_p.begin(), n2_p.end());
  return params;
}

std::shared_ptr<Tensor>
TransformerDecoderLayer::forward(const std::shared_ptr<Tensor> &input) {
  return forward(input, nullptr);
}

std::shared_ptr<Tensor>
TransformerDecoderLayer::forward(const std::shared_ptr<Tensor> &input,
                                 KVCache *cache) {
  auto attn_out = self_attn_->forward(input, cache);
  auto dropped_attn = dropout1_->forward(attn_out);
  auto x1 = norm1_->forward(input->add(dropped_attn));

  size_t B = x1->shape()[0];
  size_t S = x1->shape()[1];
  size_t E = x1->shape()[2];

  auto x1_flat = x1->reshape(Shape{B * S, E});
  auto ffn_flat = ffn_->forward(x1_flat);
  auto ffn_out = ffn_flat->reshape(Shape{B, S, E});

  auto dropped_ffn = dropout2_->forward(ffn_out);
  return norm2_->forward(x1->add(dropped_ffn));
}

TransformerDecoder::TransformerDecoder(size_t vocab_size, size_t embed_dim,
                                       size_t num_heads, size_t dim_feedforward,
                                       size_t num_layers, size_t max_seq_len,
                                       ExecutionBackend *backend)
    : max_seq_len_(max_seq_len) {
  token_emb_ = std::make_shared<Embedding>(vocab_size, embed_dim, backend);

  for (size_t i = 0; i < num_layers; ++i) {
    layers_.push_back(std::make_shared<TransformerDecoderLayer>(
        embed_dim, num_heads, dim_feedforward, 0.1f, backend));
  }

  lm_head_ = std::make_shared<Dense>(embed_dim, vocab_size, backend);
}

void TransformerDecoder::set_training(bool training) {
  Layer::set_training(training);
  token_emb_->set_training(training);
  for (auto &layer : layers_) {
    layer->set_training(training);
  }
  lm_head_->set_training(training);
}

std::vector<std::shared_ptr<Tensor>> TransformerDecoder::parameters() {
  std::vector<std::shared_ptr<Tensor>> params;
  auto t_p = token_emb_->parameters();
  params.insert(params.end(), t_p.begin(), t_p.end());

  for (auto &layer : layers_) {
    auto l_p = layer->parameters();
    params.insert(params.end(), l_p.begin(), l_p.end());
  }

  auto h_p = lm_head_->parameters();
  params.insert(params.end(), h_p.begin(), h_p.end());

  return params;
}

std::shared_ptr<Tensor>
TransformerDecoder::forward(const std::shared_ptr<Tensor> &input) {
  return forward(input, nullptr);
}

std::shared_ptr<Tensor>
TransformerDecoder::forward(const std::shared_ptr<Tensor> &input,
                            std::vector<KVCache> *caches) {
  if (input->shape().size() != 2) {
    throw std::runtime_error(
        "TransformerDecoder input must be of shape (B, S)");
  }
  size_t B = input->shape()[0];
  size_t S = input->shape()[1];

  size_t S_past = 0;
  if (caches && !caches->empty() && caches->at(0).k) {
    S_past = caches->at(0).k->shape()[2];
  }

  if (S_past + S > max_seq_len_) {
    throw std::runtime_error(
        "Sequence length S plus past cached length exceeds max_seq_len");
  }

  auto token_embed = token_emb_->forward(input);
  auto x = token_embed;

  for (size_t l = 0; l < layers_.size(); ++l) {
    KVCache *layer_cache = caches ? &((*caches)[l]) : nullptr;
    x = layers_[l]->forward(x, layer_cache);
  }

  size_t E = x->shape()[2];
  auto x_flat = x->reshape(Shape{B * S, E});
  auto logits_flat = lm_head_->forward(x_flat);
  return logits_flat->reshape(Shape{B, S, logits_flat->shape()[1]});
}

std::vector<size_t>
TransformerDecoder::generate(const std::vector<size_t> &prompt,
                             size_t max_new_tokens) {
  std::vector<size_t> generated = prompt;
  auto backend = token_emb_->parameters()[0]->backend();

  bool was_training = training_;
  set_training(false);

  for (size_t step = 0; step < max_new_tokens; ++step) {
    size_t S = generated.size();
    size_t start_idx = 0;
    if (S > max_seq_len_) {
      start_idx = S - max_seq_len_;
      S = max_seq_len_;
    }

    std::vector<float> input_data(S);
    for (size_t i = 0; i < S; ++i) {
      input_data[i] = static_cast<float>(generated[start_idx + i]);
    }

    auto input_tensor =
        std::make_shared<Tensor>(Shape{1, S}, input_data, backend);
    auto logits = forward(input_tensor);

    std::vector<float> host_logits = logits->to_host();
    size_t vocab_size = logits->shape()[2];

    size_t last_token_offset = (S - 1) * vocab_size;
    float max_val = host_logits[last_token_offset];
    size_t argmax = 0;
    for (size_t v = 1; v < vocab_size; ++v) {
      if (host_logits[last_token_offset + v] > max_val) {
        max_val = host_logits[last_token_offset + v];
        argmax = v;
      }
    }
    generated.push_back(argmax);
  }

  set_training(was_training);
  return generated;
}

std::vector<size_t> TransformerDecoder::generate_advanced(
    const std::vector<size_t> &prompt, size_t max_new_tokens, float temperature,
    size_t top_k, float top_p, unsigned int seed, bool use_kv_cache) {
  std::vector<size_t> generated = prompt;
  auto backend = token_emb_->parameters()[0]->backend();

  bool was_training = training_;
  set_training(false);

  std::mt19937 gen(seed);

  std::vector<KVCache> caches;
  if (use_kv_cache) {
    caches.resize(layers_.size());
  }

  for (size_t step = 0; step < max_new_tokens; ++step) {
    size_t S = generated.size();

    std::shared_ptr<Tensor> input_tensor;
    if (use_kv_cache && step > 0) {
      std::vector<float> input_data = {static_cast<float>(generated.back())};
      input_tensor = std::make_shared<Tensor>(Shape{1, 1}, input_data, backend);
    } else {
      size_t start_idx = 0;
      if (S > max_seq_len_) {
        start_idx = S - max_seq_len_;
        S = max_seq_len_;
      }
      std::vector<float> input_data(S);
      for (size_t i = 0; i < S; ++i) {
        input_data[i] = static_cast<float>(generated[start_idx + i]);
      }
      input_tensor = std::make_shared<Tensor>(Shape{1, S}, input_data, backend);
    }

    std::shared_ptr<Tensor> logits;
    if (use_kv_cache) {
      logits = forward(input_tensor, &caches);
    } else {
      logits = forward(input_tensor, nullptr);
    }

    std::vector<float> host_logits = logits->to_host();
    size_t vocab_size = logits->shape()[2];
    size_t S_out = logits->shape()[1];
    size_t last_token_offset = (S_out - 1) * vocab_size;

    std::vector<float> token_logits(vocab_size);
    for (size_t v = 0; v < vocab_size; ++v) {
      token_logits[v] = host_logits[last_token_offset + v];
    }

    size_t next_token = 0;

    if (temperature <= 0.0f) {
      float max_val = token_logits[0];
      for (size_t v = 1; v < vocab_size; ++v) {
        if (token_logits[v] > max_val) {
          max_val = token_logits[v];
          next_token = v;
        }
      }
    } else {
      for (size_t v = 0; v < vocab_size; ++v) {
        token_logits[v] /= temperature;
      }

      float max_logit = token_logits[0];
      for (size_t v = 1; v < vocab_size; ++v) {
        if (token_logits[v] > max_logit) {
          max_logit = token_logits[v];
        }
      }

      std::vector<float> probs(vocab_size);
      float sum_probs = 0.0f;
      for (size_t v = 0; v < vocab_size; ++v) {
        probs[v] = std::exp(token_logits[v] - max_logit);
        sum_probs += probs[v];
      }
      for (size_t v = 0; v < vocab_size; ++v) {
        probs[v] /= sum_probs;
      }

      std::vector<std::pair<float, size_t>> indexed_probs(vocab_size);
      for (size_t v = 0; v < vocab_size; ++v) {
        indexed_probs[v] = {probs[v], v};
      }

      std::sort(indexed_probs.begin(), indexed_probs.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });

      if (top_k > 0 && top_k < vocab_size) {
        for (size_t i = top_k; i < vocab_size; ++i) {
          indexed_probs[i].first = 0.0f;
        }
      }

      if (top_p > 0.0f && top_p < 1.0f) {
        float cum_sum = 0.0f;
        bool cut = false;
        for (size_t i = 0; i < vocab_size; ++i) {
          if (cut) {
            indexed_probs[i].first = 0.0f;
          } else {
            cum_sum += indexed_probs[i].first;
            if (cum_sum >= top_p) {
              cut = true;
            }
          }
        }
      }

      float final_sum = 0.0f;
      for (size_t v = 0; v < vocab_size; ++v) {
        final_sum += indexed_probs[v].first;
      }

      if (final_sum <= 0.0f) {
        next_token = indexed_probs[0].second;
      } else {
        std::vector<double> normalized_probs(vocab_size, 0.0);
        for (size_t v = 0; v < vocab_size; ++v) {
          size_t orig_idx = indexed_probs[v].second;
          normalized_probs[orig_idx] =
              static_cast<double>(indexed_probs[v].first / final_sum);
        }

        std::discrete_distribution<size_t> dist(normalized_probs.begin(),
                                                normalized_probs.end());
        next_token = dist(gen);
      }
    }

    generated.push_back(next_token);
  }

  set_training(was_training);
  return generated;
}

std::vector<size_t> TransformerDecoder::generate_speculative(
    TransformerDecoder *draft_model, const std::vector<size_t> &prompt,
    size_t max_new_tokens, size_t lookahead, float temperature, size_t top_k,
    float top_p, unsigned int seed, bool quantized_cache) {
  std::vector<size_t> generated = prompt;
  auto backend = token_emb_->parameters()[0]->backend();

  bool was_training = training_;
  bool draft_was_training = draft_model->training_;
  set_training(false);
  draft_model->set_training(false);

  std::mt19937 gen(seed);

  std::vector<KVCache> target_caches(layers_.size(), KVCache(quantized_cache));
  std::vector<KVCache> draft_caches(draft_model->layers_.size(),
                                    KVCache(quantized_cache));

  size_t S = generated.size();
  std::vector<float> input_data(S);
  for (size_t i = 0; i < S; ++i) {
    input_data[i] = static_cast<float>(generated[i]);
  }
  auto input_tensor =
      std::make_shared<Tensor>(Shape{1, S}, input_data, backend);
  auto target_logits = forward(input_tensor, &target_caches);
  auto draft_logits = draft_model->forward(input_tensor, &draft_caches);

  auto get_probs = [&](const std::vector<float> &logits) -> std::vector<float> {
    size_t vocab = logits.size();
    std::vector<float> probs(vocab);
    if (temperature <= 0.0f) {
      float max_val = logits[0];
      size_t argmax = 0;
      for (size_t v = 1; v < vocab; ++v) {
        if (logits[v] > max_val) {
          max_val = logits[v];
          argmax = v;
        }
      }
      probs[argmax] = 1.0f;
      return probs;
    }

    std::vector<float> scaled_logits = logits;
    for (size_t v = 0; v < vocab; ++v) {
      scaled_logits[v] /= temperature;
    }

    float max_logit = scaled_logits[0];
    for (size_t v = 1; v < vocab; ++v) {
      if (scaled_logits[v] > max_logit) {
        max_logit = scaled_logits[v];
      }
    }

    float sum_probs = 0.0f;
    for (size_t v = 0; v < vocab; ++v) {
      probs[v] = std::exp(scaled_logits[v] - max_logit);
      sum_probs += probs[v];
    }
    for (size_t v = 0; v < vocab; ++v) {
      probs[v] /= sum_probs;
    }

    std::vector<std::pair<float, size_t>> indexed_probs(vocab);
    for (size_t v = 0; v < vocab; ++v) {
      indexed_probs[v] = {probs[v], v};
    }

    std::sort(indexed_probs.begin(), indexed_probs.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    if (top_k > 0 && top_k < vocab) {
      for (size_t i = top_k; i < vocab; ++i) {
        indexed_probs[i].first = 0.0f;
      }
    }

    if (top_p > 0.0f && top_p < 1.0f) {
      float cum_sum = 0.0f;
      bool cut = false;
      for (size_t i = 0; i < vocab; ++i) {
        if (cut) {
          indexed_probs[i].first = 0.0f;
        } else {
          cum_sum += indexed_probs[i].first;
          if (cum_sum >= top_p) {
            cut = true;
          }
        }
      }
    }

    float final_sum = 0.0f;
    for (size_t v = 0; v < vocab; ++v) {
      final_sum += indexed_probs[v].first;
    }

    std::vector<float> final_probs(vocab, 0.0f);
    if (final_sum <= 0.0f) {
      final_probs[indexed_probs[0].second] = 1.0f;
    } else {
      for (size_t v = 0; v < vocab; ++v) {
        size_t orig_idx = indexed_probs[v].second;
        final_probs[orig_idx] = indexed_probs[v].first / final_sum;
      }
    }
    return final_probs;
  };

  auto sample_token = [&](const std::vector<float> &probs) -> size_t {
    std::vector<double> normalized_probs(probs.begin(), probs.end());
    std::discrete_distribution<size_t> dist(normalized_probs.begin(),
                                            normalized_probs.end());
    return dist(gen);
  };

  std::vector<float> host_t_logits = target_logits->to_host();
  size_t vocab_size = target_logits->shape()[2];
  size_t S_out = target_logits->shape()[1];
  size_t last_token_offset = (S_out - 1) * vocab_size;
  std::vector<float> first_token_logits(vocab_size);
  for (size_t v = 0; v < vocab_size; ++v) {
    first_token_logits[v] = host_t_logits[last_token_offset + v];
  }
  auto first_probs = get_probs(first_token_logits);
  size_t first_token = sample_token(first_probs);
  generated.push_back(first_token);

  size_t num_generated = 1;

  while (num_generated < max_new_tokens) {
    size_t M = std::min(lookahead, max_new_tokens - num_generated);
    if (M == 0) {
      break;
    }

    size_t target_S_start = 0;
    if (target_caches[0].quantized) {
      if (!target_caches[0].quantized_k.empty()) {
        target_S_start = target_caches[0].shape[2];
      }
    } else if (target_caches[0].k) {
      target_S_start = target_caches[0].k->shape()[2];
    }

    size_t draft_S_start = 0;
    if (draft_caches[0].quantized) {
      if (!draft_caches[0].quantized_k.empty()) {
        draft_S_start = draft_caches[0].shape[2];
      }
    } else if (draft_caches[0].k) {
      draft_S_start = draft_caches[0].k->shape()[2];
    }

    std::vector<size_t> draft_candidates;
    std::vector<std::vector<float>> draft_probs;

    for (size_t j = 0; j < M; ++j) {
      size_t next_input = (j == 0) ? generated.back() : draft_candidates.back();
      std::vector<float> d_input_data = {static_cast<float>(next_input)};
      auto d_input =
          std::make_shared<Tensor>(Shape{1, 1}, d_input_data, backend);
      auto d_logits = draft_model->forward(d_input, &draft_caches);

      std::vector<float> host_d_logits = d_logits->to_host();
      size_t d_vocab_size = d_logits->shape()[2];
      std::vector<float> d_token_logits(d_vocab_size);
      for (size_t v = 0; v < d_vocab_size; ++v) {
        d_token_logits[v] = host_d_logits[v];
      }
      auto q_j = get_probs(d_token_logits);
      size_t candidate = sample_token(q_j);
      draft_candidates.push_back(candidate);
      draft_probs.push_back(q_j);
    }

    std::vector<float> t_input_data(M + 1);
    t_input_data[0] = static_cast<float>(generated.back());
    for (size_t i = 0; i < M; ++i) {
      t_input_data[i + 1] = static_cast<float>(draft_candidates[i]);
    }
    auto t_input =
        std::make_shared<Tensor>(Shape{1, M + 1}, t_input_data, backend);
    auto t_logits = forward(t_input, &target_caches);

    std::vector<float> host_t_logits_step = t_logits->to_host();
    size_t t_vocab_size = t_logits->shape()[2];
    std::vector<std::vector<float>> target_probs(M + 1);
    for (size_t i = 0; i <= M; ++i) {
      size_t offset = i * t_vocab_size;
      std::vector<float> step_logits(t_vocab_size);
      for (size_t v = 0; v < t_vocab_size; ++v) {
        step_logits[v] = host_t_logits_step[offset + v];
      }
      target_probs[i] = get_probs(step_logits);
    }

    bool all_accepted = true;
    size_t rejected_idx = 0;

    for (size_t j = 0; j < M; ++j) {
      size_t tilde_x = draft_candidates[j];
      float q = draft_probs[j][tilde_x];
      float p = target_probs[j][tilde_x];

      bool accepted = false;
      if (temperature <= 0.0f) {
        size_t target_greedy = std::distance(
            target_probs[j].begin(),
            std::max_element(target_probs[j].begin(), target_probs[j].end()));
        accepted = (tilde_x == target_greedy);
      } else {
        std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
        float r = uniform_dist(gen);
        accepted = (r * q < p);
      }

      if (accepted) {
        generated.push_back(tilde_x);
        num_generated++;
      } else {
        all_accepted = false;
        rejected_idx = j;
        break;
      }
    }

    if (!all_accepted) {
      std::vector<float> probs_rec(t_vocab_size, 0.0f);
      float sum_rec = 0.0f;
      for (size_t v = 0; v < t_vocab_size; ++v) {
        probs_rec[v] = std::max(0.0f, target_probs[rejected_idx][v] -
                                          draft_probs[rejected_idx][v]);
        sum_rec += probs_rec[v];
      }
      size_t recovered_token = 0;
      if (sum_rec <= 0.0f) {
        recovered_token = sample_token(target_probs[rejected_idx]);
      } else {
        for (size_t v = 0; v < t_vocab_size; ++v) {
          probs_rec[v] /= sum_rec;
        }
        recovered_token = sample_token(probs_rec);
      }

      generated.push_back(recovered_token);
      num_generated++;

      for (auto &cache : target_caches) {
        cache.truncate(target_S_start + rejected_idx + 1);
      }
      for (auto &cache : draft_caches) {
        cache.truncate(draft_S_start + rejected_idx + 1);
      }

      std::vector<float> catchup_data = {static_cast<float>(recovered_token)};
      auto catchup_input =
          std::make_shared<Tensor>(Shape{1, 1}, catchup_data, backend);
      draft_model->forward(catchup_input, &draft_caches);
    } else {
      if (num_generated < max_new_tokens) {
        size_t final_token = sample_token(target_probs[M]);
        generated.push_back(final_token);
        num_generated++;

        std::vector<float> catchup_data = {
            static_cast<float>(draft_candidates[M - 1])};
        auto catchup_input =
            std::make_shared<Tensor>(Shape{1, 1}, catchup_data, backend);
        draft_model->forward(catchup_input, &draft_caches);
      }
    }
  }

  set_training(was_training);
  draft_model->set_training(draft_was_training);
  return generated;
}

StochasticDepth::StochasticDepth(float drop_prob) : drop_prob_(drop_prob) {
  if (drop_prob_ < 0.0f || drop_prob_ > 1.0f) {
    throw std::runtime_error("drop_prob must be between 0.0 and 1.0");
  }
}

std::shared_ptr<Tensor>
StochasticDepth::forward(const std::shared_ptr<Tensor> &input) {
  return input;
}

std::shared_ptr<Tensor>
StochasticDepth::forward_residual(const std::shared_ptr<Tensor> &x,
                                  const std::shared_ptr<Tensor> &sublayer_out) {
  if (!training_ || drop_prob_ <= 0.0f) {
    return x->add(sublayer_out);
  }
  if (drop_prob_ >= 1.0f) {
    return x;
  }

  static std::mt19937 gen(1337);
  std::bernoulli_distribution dist(1.0f - drop_prob_);
  bool survive = dist(gen);

  if (!survive) {
    return x;
  }

  float survival_prob = 1.0f - drop_prob_;
  auto scaled = sublayer_out * (1.0f / survival_prob);
  return x->add(scaled);
}

SwiGLU::SwiGLU(size_t in_features, size_t out_features,
               ExecutionBackend *backend) {
  gate_proj_ = std::make_shared<Dense>(in_features, out_features, backend);
  up_proj_ = std::make_shared<Dense>(in_features, out_features, backend);
  down_proj_ = std::make_shared<Dense>(out_features, in_features, backend);
}

std::shared_ptr<Tensor> SwiGLU::forward(const std::shared_ptr<Tensor> &input) {
  auto gate = gate_proj_->forward(input);
  auto up = up_proj_->forward(input);
  auto activated = gate->mul(gate->sigmoid());
  auto gated = activated->mul(up);
  return down_proj_->forward(gated);
}

std::vector<std::shared_ptr<Tensor>> SwiGLU::parameters() {
  std::vector<std::shared_ptr<Tensor>> params;
  auto g_p = gate_proj_->parameters();
  auto u_p = up_proj_->parameters();
  auto d_p = down_proj_->parameters();
  params.insert(params.end(), g_p.begin(), g_p.end());
  params.insert(params.end(), u_p.begin(), u_p.end());
  params.insert(params.end(), d_p.begin(), d_p.end());
  return params;
}

void SwiGLU::set_training(bool training) {
  Layer::set_training(training);
  gate_proj_->set_training(training);
  up_proj_->set_training(training);
  down_proj_->set_training(training);
}
