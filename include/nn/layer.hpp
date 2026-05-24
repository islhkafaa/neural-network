#ifndef LAYER_HPP
#define LAYER_HPP

#include "core/tensor.hpp"
#include <memory>
#include <vector>

class SwiGLU;
struct LoRAConfig;
class LoRALinear;

class Layer {
public:
  virtual ~Layer() = default;
  virtual std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) = 0;
  virtual void set_training(bool training) { training_ = training; }
  virtual std::vector<std::shared_ptr<Tensor>> parameters() { return {}; }

  void save_weights(const std::string &filepath);
  void load_weights(const std::string &filepath);

protected:
  bool training_ = true;
};

class ReLU : public Layer {
public:
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override {
    return input->relu();
  }
};

class Sigmoid : public Layer {
public:
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override {
    return input->sigmoid();
  }
};

class Tanh : public Layer {
public:
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override {
    return input->tanh();
  }
};

class Softmax : public Layer {
public:
  explicit Softmax(size_t axis) : axis_(axis) {}
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override {
    return input->softmax(axis_);
  }

private:
  size_t axis_;
};

class Dense : public Layer {
public:
  Dense(size_t in_features, size_t out_features,
        ExecutionBackend *backend = nullptr);
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
  std::vector<std::shared_ptr<Tensor>> parameters() override {
    return {weight_, bias_};
  }

private:
  std::shared_ptr<Tensor> weight_;
  std::shared_ptr<Tensor> bias_;
};

class Dropout : public Layer {
public:
  explicit Dropout(float p = 0.5f) : p_(p) {}
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;

private:
  float p_;
};

class LayerNorm : public Layer {
public:
  LayerNorm(const Shape &normalized_shape, ExecutionBackend *backend = nullptr,
            float eps = 1e-5f);
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
  std::vector<std::shared_ptr<Tensor>> parameters() override {
    return {gamma_, beta_};
  }

private:
  Shape normalized_shape_;
  float eps_;
  std::shared_ptr<Tensor> gamma_;
  std::shared_ptr<Tensor> beta_;
};

class Embedding : public Layer {
public:
  Embedding(size_t num_embeddings, size_t embedding_dim,
            ExecutionBackend *backend = nullptr);
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
  std::vector<std::shared_ptr<Tensor>> parameters() override {
    return {weight_};
  }

private:
  std::shared_ptr<Tensor> weight_;
};

class Conv2D : public Layer {
public:
  Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size,
         size_t stride = 1, size_t padding = 0,
         ExecutionBackend *backend = nullptr);
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
  std::vector<std::shared_ptr<Tensor>> parameters() override {
    return {weight_, bias_};
  }

private:
  std::shared_ptr<Tensor> weight_;
  std::shared_ptr<Tensor> bias_;
  size_t stride_;
  size_t padding_;
};

class MaxPooling2D : public Layer {
public:
  explicit MaxPooling2D(size_t pool_size, size_t stride = 1)
      : pool_size_(pool_size), stride_(stride) {}
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;

private:
  size_t pool_size_;
  size_t stride_;
};

class RNN : public Layer {
public:
  RNN(size_t input_size, size_t hidden_size,
      ExecutionBackend *backend = nullptr);
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
  std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input,
                                  const std::shared_ptr<Tensor> &h0);
  std::vector<std::shared_ptr<Tensor>> parameters() override {
    return {W_ih_, W_hh_, b_ih_, b_hh_};
  }

private:
  size_t hidden_size_;
  std::shared_ptr<Tensor> W_ih_;
  std::shared_ptr<Tensor> W_hh_;
  std::shared_ptr<Tensor> b_ih_;
  std::shared_ptr<Tensor> b_hh_;
};

class LSTM : public Layer {
public:
  LSTM(size_t input_size, size_t hidden_size,
       ExecutionBackend *backend = nullptr);
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
  std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> forward(
      const std::shared_ptr<Tensor> &input,
      const std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> &hx);
  std::vector<std::shared_ptr<Tensor>> parameters() override {
    return {W_ii_, W_hi_, b_ii_, b_hi_, W_if_, W_hf_, b_if_, b_hf_,
            W_ig_, W_hg_, b_ig_, b_hg_, W_io_, W_ho_, b_io_, b_ho_};
  }

private:
  size_t hidden_size_;
  std::shared_ptr<Tensor> W_ii_, W_hi_, b_ii_, b_hi_;
  std::shared_ptr<Tensor> W_if_, W_hf_, b_if_, b_hf_;
  std::shared_ptr<Tensor> W_ig_, W_hg_, b_ig_, b_hg_;
  std::shared_ptr<Tensor> W_io_, W_ho_, b_io_, b_ho_;
};

class Sequential : public Layer {
public:
  Sequential() = default;
  explicit Sequential(std::vector<std::shared_ptr<Layer>> layers)
      : layers_(layers) {}
  void add(std::shared_ptr<Layer> layer) { layers_.push_back(layer); }

  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
  void set_training(bool training) override;
  std::vector<std::shared_ptr<Tensor>> parameters() override;

private:
  std::vector<std::shared_ptr<Layer>> layers_;
};

struct KVCache {
  std::shared_ptr<Tensor> k;
  std::shared_ptr<Tensor> v;
};

class MultiHeadAttention : public Layer {
public:
  MultiHeadAttention(size_t embed_dim, size_t num_heads, bool causal = false,
                     ExecutionBackend *backend = nullptr);
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
  std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input,
                                  KVCache *cache);
  std::vector<std::shared_ptr<Tensor>> parameters() override;

  void apply_lora(const LoRAConfig &cfg);
  std::vector<std::shared_ptr<LoRALinear>> lora_modules() const;

private:
  size_t num_heads_;
  size_t head_dim_;
  bool causal_;
  std::shared_ptr<Layer> q_proj_;
  std::shared_ptr<Layer> k_proj_;
  std::shared_ptr<Layer> v_proj_;
  std::shared_ptr<Layer> out_proj_;
};

class TransformerEncoderLayer : public Layer {
public:
  TransformerEncoderLayer(size_t embed_dim, size_t num_heads,
                          size_t dim_feedforward, float dropout_p = 0.1f,
                          ExecutionBackend *backend = nullptr);
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
  std::vector<std::shared_ptr<Tensor>> parameters() override;
  void set_training(bool training) override;

private:
  std::shared_ptr<MultiHeadAttention> self_attn_;
  std::shared_ptr<Dense> linear1_;
  std::shared_ptr<Dense> linear2_;
  std::shared_ptr<LayerNorm> norm1_;
  std::shared_ptr<LayerNorm> norm2_;
  std::shared_ptr<Dropout> dropout1_;
  std::shared_ptr<Dropout> dropout2_;
};

class TransformerDecoderLayer : public Layer {
public:
  TransformerDecoderLayer(size_t embed_dim, size_t num_heads,
                          size_t dim_feedforward, float dropout_p = 0.1f,
                          ExecutionBackend *backend = nullptr);
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
  std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input,
                                  KVCache *cache);
  std::vector<std::shared_ptr<Tensor>> parameters() override;
  void set_training(bool training) override;

private:
  std::shared_ptr<MultiHeadAttention> self_attn_;
  std::shared_ptr<SwiGLU> ffn_;
  std::shared_ptr<LayerNorm> norm1_;
  std::shared_ptr<LayerNorm> norm2_;
  std::shared_ptr<Dropout> dropout1_;
  std::shared_ptr<Dropout> dropout2_;
};

class TransformerDecoder : public Layer {
public:
  TransformerDecoder(size_t vocab_size, size_t embed_dim, size_t num_heads,
                     size_t dim_feedforward, size_t num_layers,
                     size_t max_seq_len, ExecutionBackend *backend = nullptr);
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
  std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input,
                                  std::vector<KVCache> *caches);
  std::vector<std::shared_ptr<Tensor>> parameters() override;
  void set_training(bool training) override;
  std::vector<size_t> generate(const std::vector<size_t> &prompt,
                               size_t max_new_tokens);
  std::vector<size_t> generate_advanced(const std::vector<size_t> &prompt,
                                        size_t max_new_tokens,
                                        float temperature = 1.0f,
                                        size_t top_k = 0, float top_p = 1.0f,
                                        unsigned int seed = 42,
                                        bool use_kv_cache = true);

private:
  size_t max_seq_len_;
  std::shared_ptr<Embedding> token_emb_;
  std::vector<std::shared_ptr<TransformerDecoderLayer>> layers_;
  std::shared_ptr<Dense> lm_head_;
};

class SwiGLU : public Layer {
public:
  SwiGLU(size_t in_features, size_t out_features,
         ExecutionBackend *backend = nullptr);
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
  std::vector<std::shared_ptr<Tensor>> parameters() override;
  void set_training(bool training) override;

private:
  std::shared_ptr<Dense> gate_proj_;
  std::shared_ptr<Dense> up_proj_;
  std::shared_ptr<Dense> down_proj_;
};

class StochasticDepth : public Layer {
public:
  explicit StochasticDepth(float drop_prob = 0.2f);
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
  std::shared_ptr<Tensor>
  forward_residual(const std::shared_ptr<Tensor> &x,
                   const std::shared_ptr<Tensor> &sublayer_out);

private:
  float drop_prob_;
};

#endif // LAYER_HPP
