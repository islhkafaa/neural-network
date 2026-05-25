#include "core/tensor.hpp"
#include "nn/layer.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

TEST(TransformerTest, ReshapeForwardBackward) {
  // Create a 2x3 tensor
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto t = std::make_shared<Tensor>(Shape{2, 3}, data);
  t->set_requires_grad(true);

  auto r = t->reshape(Shape{3, 2});
  EXPECT_EQ(r->shape(), (Shape{3, 2}));

  std::vector<float> expected_r = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  EXPECT_EQ(r->to_host(), expected_r);

  // Backward check
  auto loss = r->sum({});
  loss->backward();

  EXPECT_EQ(t->grad()->shape(), (Shape{2, 3}));
  std::vector<float> expected_grad(6, 1.0f);
  EXPECT_EQ(t->grad()->to_host(), expected_grad);
}

TEST(TransformerTest, TransposeForwardBackward) {
  // Create a 3D tensor of shape 2x3x4
  std::vector<float> data(24);
  for (size_t i = 0; i < 24; ++i) {
    data[i] = static_cast<float>(i + 1);
  }
  auto t = std::make_shared<Tensor>(Shape{2, 3, 4}, data);
  t->set_requires_grad(true);

  // Transpose dim 1 and 2 -> shape should be 2x4x3
  auto tr = t->transpose(1, 2);
  EXPECT_EQ(tr->shape(), (Shape{2, 4, 3}));

  // Verify transposed output elements
  std::vector<float> host_tr = tr->to_host();
  // Element at (b, j, i) in tr should be equal to element at (b, i, j) in t
  for (size_t b = 0; b < 2; ++b) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        float val_orig = data[b * 12 + i * 4 + j];
        float val_trans = host_tr[b * 12 + j * 3 + i];
        EXPECT_FLOAT_EQ(val_orig, val_trans);
      }
    }
  }

  // Backward check
  auto loss = tr->sum({});
  loss->backward();

  EXPECT_EQ(t->grad()->shape(), (Shape{2, 3, 4}));
  std::vector<float> expected_grad(24, 1.0f);
  EXPECT_EQ(t->grad()->to_host(), expected_grad);
}

TEST(TransformerTest, BatchedMatrixMultiplication) {
  // Tensor A of shape (2, 2, 3, 4)
  std::vector<float> data_a(48, 1.0f);
  auto a = std::make_shared<Tensor>(Shape{2, 2, 3, 4}, data_a);
  a->set_requires_grad(true);

  // Tensor B of shape (2, 2, 4, 2)
  std::vector<float> data_b(32, 2.0f);
  auto b = std::make_shared<Tensor>(Shape{2, 2, 4, 2}, data_b);
  b->set_requires_grad(true);

  auto out = a->bmm(b);
  EXPECT_EQ(out->shape(), (Shape{2, 2, 3, 2}));

  // Mathematical check: since A is all 1s and B is all 2s, and inner dimension
  // is 4: Each output cell is sum_{k=1..4}(1 * 2) = 8.0f
  std::vector<float> host_out = out->to_host();
  for (float val : host_out) {
    EXPECT_FLOAT_EQ(val, 8.0f);
  }

  // Backward check
  auto loss = out->sum({});
  loss->backward();

  EXPECT_EQ(a->grad()->shape(), (Shape{2, 2, 3, 4}));
  EXPECT_EQ(b->grad()->shape(), (Shape{2, 2, 4, 2}));

  // Verify numerical values of gradients:
  // grad_a = grad_out * B^T
  // since grad_out is all 1s, grad_a cell at index k is sum_{n=1..2}(1 *
  // B_{k,n}) = 2 + 2 = 4.0f
  for (float val : a->grad()->to_host()) {
    EXPECT_FLOAT_EQ(val, 4.0f);
  }
}

TEST(TransformerTest, MultiHeadAttentionForwardBackward) {
  size_t B = 2;
  size_t S = 4;
  size_t E = 8;
  size_t H = 2;

  std::vector<float> input_data(B * S * E, 0.1f);
  auto input = std::make_shared<Tensor>(Shape{B, S, E}, input_data);
  input->set_requires_grad(true);

  MultiHeadAttention mha(E, H);
  auto output = mha.forward(input);

  EXPECT_EQ(output->shape(), (Shape{B, S, E}));

  // Perform backward pass to verify autograd tape traverses all weights
  auto loss = output->sum({});
  loss->backward();

  EXPECT_EQ(input->grad()->shape(), (Shape{B, S, E}));
  for (auto &param : mha.parameters()) {
    ASSERT_TRUE(param->grad() != nullptr);
    EXPECT_EQ(param->grad()->shape(), param->shape());
  }
}

TEST(TransformerTest, TransformerEncoderLayerForwardBackward) {
  size_t B = 2;
  size_t S = 4;
  size_t E = 8;
  size_t H = 2;
  size_t FFN_DIM = 16;

  std::vector<float> input_data(B * S * E, 0.5f);
  auto input = std::make_shared<Tensor>(Shape{B, S, E}, input_data);
  input->set_requires_grad(true);

  TransformerEncoderLayer encoder(E, H, FFN_DIM);
  encoder.set_training(true);

  auto output = encoder.forward(input);
  EXPECT_EQ(output->shape(), (Shape{B, S, E}));

  auto loss = output->sum({});
  loss->backward();

  EXPECT_EQ(input->grad()->shape(), (Shape{B, S, E}));
  for (auto &param : encoder.parameters()) {
    ASSERT_TRUE(param->grad() != nullptr);
    EXPECT_EQ(param->grad()->shape(), param->shape());
  }
}

TEST(TransformerTest, CausalMultiHeadAttention) {
  size_t B = 1;
  size_t S = 3;
  size_t E = 4;
  size_t H = 2;

  std::vector<float> data1 = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                              0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
  std::vector<float> data2 = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                              0.7f, 0.8f, 9.9f, 9.9f, 9.9f, 9.9f};

  auto input1 = std::make_shared<Tensor>(Shape{B, S, E}, data1);
  auto input2 = std::make_shared<Tensor>(Shape{B, S, E}, data2);

  MultiHeadAttention mha(E, H, true);
  mha.set_training(false);

  auto out1 = mha.forward(input1)->to_host();
  auto out2 = mha.forward(input2)->to_host();

  for (size_t i = 0; i < 2 * E; ++i) {
    EXPECT_FLOAT_EQ(out1[i], out2[i]);
  }
}

TEST(TransformerTest, TransformerDecoderLayer) {
  size_t B = 2;
  size_t S = 4;
  size_t E = 8;
  size_t H = 2;
  size_t FFN_DIM = 16;

  std::vector<float> input_data(B * S * E, 0.5f);
  auto input = std::make_shared<Tensor>(Shape{B, S, E}, input_data);
  input->set_requires_grad(true);

  TransformerDecoderLayer decoder(E, H, FFN_DIM);
  decoder.set_training(true);

  auto output = decoder.forward(input);
  EXPECT_EQ(output->shape(), (Shape{B, S, E}));

  auto loss = output->sum({});
  loss->backward();

  EXPECT_EQ(input->grad()->shape(), (Shape{B, S, E}));
  for (auto &param : decoder.parameters()) {
    ASSERT_TRUE(param->grad() != nullptr);
    EXPECT_EQ(param->grad()->shape(), param->shape());
  }
}

TEST(TransformerTest, TransformerDecoderEndToEndTrainingAndGeneration) {
  size_t vocab_size = 5;
  size_t embed_dim = 8;
  size_t num_heads = 2;
  size_t dim_feedforward = 16;
  size_t num_layers = 2;
  size_t max_seq_len = 8;

  TransformerDecoder model(vocab_size, embed_dim, num_heads, dim_feedforward,
                           num_layers, max_seq_len);

  std::vector<size_t> prompt = {1, 2, 3};
  size_t max_new_tokens = 4;
  auto generated = model.generate(prompt, max_new_tokens);

  EXPECT_EQ(generated.size(), prompt.size() + max_new_tokens);
  for (size_t tok : generated) {
    EXPECT_LT(tok, vocab_size);
  }

  model.set_training(true);

  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f,
                                   1.0f, 2.0f, 3.0f, 4.0f};
  auto input = std::make_shared<Tensor>(Shape{2, 4}, input_data);

  auto logits = model.forward(input);
  EXPECT_EQ(logits->shape(), (Shape{2, 4, vocab_size}));

  auto loss = logits->sum({});
  loss->backward();

  for (auto &param : model.parameters()) {
    ASSERT_TRUE(param->grad() != nullptr);
    EXPECT_EQ(param->grad()->shape(), param->shape());

    std::vector<float> p_data = param->to_host();
    std::vector<float> g_data = param->grad()->to_host();
    for (size_t i = 0; i < p_data.size(); ++i) {
      p_data[i] -= 0.01f * g_data[i];
    }
    param->copy_from_host(p_data);
  }
}

TEST(TransformerTest, KVCachingCorrectness) {
  size_t vocab_size = 5;
  size_t embed_dim = 8;
  size_t num_heads = 2;
  size_t dim_feedforward = 16;
  size_t num_layers = 2;
  size_t max_seq_len = 16;

  TransformerDecoder model(vocab_size, embed_dim, num_heads, dim_feedforward,
                           num_layers, max_seq_len);

  std::vector<size_t> prompt = {1, 2, 3};
  size_t max_new_tokens = 4;

  auto gen_no_cache =
      model.generate_advanced(prompt, max_new_tokens, 0.0f, 0, 1.0f, 42, false);
  auto gen_with_cache =
      model.generate_advanced(prompt, max_new_tokens, 0.0f, 0, 1.0f, 42, true);

  EXPECT_EQ(gen_no_cache.size(), gen_with_cache.size());
  for (size_t i = 0; i < gen_no_cache.size(); ++i) {
    EXPECT_EQ(gen_no_cache[i], gen_with_cache[i]);
  }
}

TEST(TransformerTest, AdvancedSamplerValidation) {
  size_t vocab_size = 10;
  size_t embed_dim = 8;
  size_t num_heads = 2;
  size_t dim_feedforward = 16;
  size_t num_layers = 2;
  size_t max_seq_len = 16;

  TransformerDecoder model(vocab_size, embed_dim, num_heads, dim_feedforward,
                           num_layers, max_seq_len);

  std::vector<size_t> prompt = {1, 2, 3};
  size_t max_new_tokens = 5;

  auto gen_seed42_a =
      model.generate_advanced(prompt, max_new_tokens, 1.0f, 3, 0.9f, 42, true);
  auto gen_seed42_b =
      model.generate_advanced(prompt, max_new_tokens, 1.0f, 3, 0.9f, 42, true);

  for (size_t i = 0; i < gen_seed42_a.size(); ++i) {
    EXPECT_EQ(gen_seed42_a[i], gen_seed42_b[i]);
  }

  auto gen_seed99 =
      model.generate_advanced(prompt, max_new_tokens, 1.0f, 3, 0.9f, 99, true);

  EXPECT_EQ(gen_seed99.size(), gen_seed42_a.size());
  for (size_t tok : gen_seed99) {
    EXPECT_LT(tok, vocab_size);
  }
}

TEST(TransformerTest, SpeculativeDecodingParity) {
  size_t vocab_size = 8;
  size_t embed_dim = 8;
  size_t num_heads = 2;
  size_t dim_feedforward = 16;
  size_t max_seq_len = 32;

  TransformerDecoder target_model(vocab_size, embed_dim, num_heads,
                                  dim_feedforward, 2, max_seq_len);
  TransformerDecoder draft_model(vocab_size, embed_dim, num_heads,
                                 dim_feedforward, 1, max_seq_len);

  // Initialize weights so they don't produce NaNs
  // Let's run a simple forward/generation to verify they work
  std::vector<size_t> prompt = {1, 2, 3};
  size_t max_new_tokens = 6;

  // Greedy Mode Parity (temperature = 0.0f)
  // Speculative decoding MUST yield identical output to standard autoregressive
  // decoding
  auto gen_ar = target_model.generate_advanced(prompt, max_new_tokens, 0.0f, 0,
                                               1.0f, 42, true);
  auto gen_spec = target_model.generate_speculative(
      &draft_model, prompt, max_new_tokens, 3, 0.0f, 0, 1.0f, 42, false);

  ASSERT_EQ(gen_ar.size(), gen_spec.size());
  for (size_t i = 0; i < gen_ar.size(); ++i) {
    EXPECT_EQ(gen_ar[i], gen_spec[i]);
  }

  // Stochastic Mode Verification (temperature = 1.0f)
  // Ensure that the speculative generation runs successfully and produces valid
  // tokens
  auto gen_spec_stochastic = target_model.generate_speculative(
      &draft_model, prompt, max_new_tokens, 3, 1.0f, 4, 0.9f, 42, false);
  EXPECT_EQ(gen_spec_stochastic.size(), prompt.size() + max_new_tokens);
  for (size_t tok : gen_spec_stochastic) {
    EXPECT_LT(tok, vocab_size);
  }
}

TEST(TransformerTest, SpeculativeDecodingQuantized) {
  size_t vocab_size = 6;
  size_t embed_dim = 8;
  size_t num_heads = 2;
  size_t dim_feedforward = 16;
  size_t max_seq_len = 32;

  TransformerDecoder target_model(vocab_size, embed_dim, num_heads,
                                  dim_feedforward, 2, max_seq_len);
  TransformerDecoder draft_model(vocab_size, embed_dim, num_heads,
                                 dim_feedforward, 1, max_seq_len);

  std::vector<size_t> prompt = {2, 4, 1};
  size_t max_new_tokens = 5;

  // Run speculative decoding with quantized KV cache
  auto gen_spec_quant = target_model.generate_speculative(
      &draft_model, prompt, max_new_tokens, 2, 0.0f, 0, 1.0f, 42, true);

  EXPECT_EQ(gen_spec_quant.size(), prompt.size() + max_new_tokens);
  for (size_t tok : gen_spec_quant) {
    EXPECT_LT(tok, vocab_size);
  }
}
