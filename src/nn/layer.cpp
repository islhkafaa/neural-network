#include "nn/layer.hpp"
#include "core/serialization.hpp"
#include "nn/initialization.hpp"
#include "nn/lora.hpp"
#include "optim/grad_scaler.hpp"
#include <algorithm>
#include <cmath>
#include <random>

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
                                       bool causal, ExecutionBackend *backend)
    : num_heads_(num_heads), head_dim_(embed_dim / num_heads), causal_(causal) {
  if (embed_dim % num_heads != 0) {
    throw std::runtime_error("embed_dim must be divisible by num_heads");
  }
  q_proj_ = std::make_shared<Dense>(embed_dim, embed_dim, backend);
  k_proj_ = std::make_shared<Dense>(embed_dim, embed_dim, backend);
  v_proj_ = std::make_shared<Dense>(embed_dim, embed_dim, backend);
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

std::shared_ptr<Tensor>
MultiHeadAttention::forward(const std::shared_ptr<Tensor> &input) {
  return forward(input, nullptr);
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
  auto k = k_flat->reshape(Shape{B, S, num_heads_, head_dim_});
  auto v = v_flat->reshape(Shape{B, S, num_heads_, head_dim_});

  auto q_t = q->transpose(1, 2);
  auto k_t = k->transpose(1, 2);
  auto v_t = v->transpose(1, 2);

  size_t S_past = 0;
  if (cache && cache->k) {
    S_past = cache->k->shape()[2];
  }

  q_t = q_t->rope(S_past);
  k_t = k_t->rope(S_past);

  std::shared_ptr<Tensor> active_k;
  std::shared_ptr<Tensor> active_v;

  if (cache) {
    if (!cache->k) {
      cache->k = k_t;
      cache->v = v_t;
    } else {
      size_t S_past = cache->k->shape()[2];
      size_t S_total = S_past + S;
      size_t D = head_dim_;

      std::vector<float> past_k = cache->k->to_host();
      std::vector<float> new_k = k_t->to_host();
      std::vector<float> concat_k(B * num_heads_ * S_total * D);

      std::vector<float> past_v = cache->v->to_host();
      std::vector<float> new_v = v_t->to_host();
      std::vector<float> concat_v(B * num_heads_ * S_total * D);

      for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < num_heads_; ++h) {
          std::copy(
              past_k.begin() + (b * num_heads_ * S_past * D + h * S_past * D),
              past_k.begin() +
                  (b * num_heads_ * S_past * D + h * S_past * D + S_past * D),
              concat_k.begin() +
                  (b * num_heads_ * S_total * D + h * S_total * D));
          std::copy(new_k.begin() + (b * num_heads_ * S * D + h * S * D),
                    new_k.begin() +
                        (b * num_heads_ * S * D + h * S * D + S * D),
                    concat_k.begin() + (b * num_heads_ * S_total * D +
                                        h * S_total * D + S_past * D));

          std::copy(
              past_v.begin() + (b * num_heads_ * S_past * D + h * S_past * D),
              past_v.begin() +
                  (b * num_heads_ * S_past * D + h * S_past * D + S_past * D),
              concat_v.begin() +
                  (b * num_heads_ * S_total * D + h * S_total * D));
          std::copy(new_v.begin() + (b * num_heads_ * S * D + h * S * D),
                    new_v.begin() +
                        (b * num_heads_ * S * D + h * S * D + S * D),
                    concat_v.begin() + (b * num_heads_ * S_total * D +
                                        h * S_total * D + S_past * D));
        }
      }

      cache->k = std::make_shared<Tensor>(Shape{B, num_heads_, S_total, D},
                                          concat_k, input->backend());
      cache->v = std::make_shared<Tensor>(Shape{B, num_heads_, S_total, D},
                                          concat_v, input->backend());
    }
    active_k = cache->k;
    active_v = cache->v;
  } else {
    active_k = k_t;
    active_v = v_t;
  }

  auto k_trans = active_k->transpose(2, 3);
  auto scores = q_t->bmm(k_trans);

  float scale = std::sqrt(static_cast<float>(head_dim_));
  auto scaled_scores = scores / scale;

  if (causal_) {
    size_t S_total = active_k->shape()[2];
    size_t S_past = S_total - S;
    auto host_data = scaled_scores->to_host();
    size_t H = num_heads_;
    for (size_t b = 0; b < B; ++b) {
      for (size_t h = 0; h < H; ++h) {
        for (size_t i = 0; i < S; ++i) {
          for (size_t j = S_past + i + 1; j < S_total; ++j) {
            size_t idx =
                b * (H * S * S_total) + h * (S * S_total) + i * S_total + j;
            float mask_val =
                (scaled_scores->dtype() == DataType::FP16) ? -65000.0f : -1e9f;
            host_data[idx] = mask_val;
          }
        }
      }
    }
    scaled_scores->copy_from_host(host_data);
  }

  auto attn = scaled_scores->softmax(3);

  auto context = attn->bmm(active_v);
  auto context_t = context->transpose(1, 2);
  auto context_flat = context_t->reshape(Shape{B * S, E});

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
