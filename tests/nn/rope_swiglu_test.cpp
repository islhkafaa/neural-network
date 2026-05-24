#include "core/tensor.hpp"
#include "nn/layer.hpp"
#include "nn/loss.hpp"
#include "optim/optimizer.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>

TEST(RopeSwigluTest, RopeAutogradVerification) {
  // Shape: [B, H, S, D] -> [1, 1, 2, 4]
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                   5.0f, 6.0f, 7.0f, 8.0f};
  auto input = std::make_shared<Tensor>(Shape{1, 1, 2, 4}, input_data);
  input->set_requires_grad(true);

  // Apply RoPE at position offset S_past = 0
  auto out = input->rope(0);
  EXPECT_EQ(out->shape(), (Shape{1, 1, 2, 4}));

  // Verify elements for S = 0 (m = 0): cos(0) = 1, sin(0) = 0. Output should be identical to input.
  std::vector<float> out_host = out->to_host();
  EXPECT_NEAR(out_host[0], 1.0f, 1e-5f);
  EXPECT_NEAR(out_host[1], 2.0f, 1e-5f);
  EXPECT_NEAR(out_host[2], 3.0f, 1e-5f);
  EXPECT_NEAR(out_host[3], 4.0f, 1e-5f);

  // Verify elements for S = 1 (m = 1):
  // D = 4.
  // For j = 0: theta_0 = 1.0. omega = 1.0. cos(1) = 0.5403023, sin(1) = 0.84147098
  // x_even = 5.0, x_odd = 6.0
  // out_even = 5.0 * cos(1) - 6.0 * sin(1) = 5.0 * 0.5403023 - 6.0 * 0.84147098 = -2.34731
  // out_odd = 5.0 * sin(1) + 6.0 * cos(1) = 5.0 * 0.84147098 + 6.0 * 0.5403023 = 7.44917
  EXPECT_NEAR(out_host[4], -2.34731f, 1e-4f);
  EXPECT_NEAR(out_host[5], 7.44917f, 1e-4f);

  // Perform backward pass to check gradient equations
  auto loss = out->sum({});
  loss->backward();

  // Gradient out is all 1s.
  // For S = 1 (m = 1), grad_out = [1, 1] for even/odd.
  // grad_in_even = g_even * cos(1) + g_odd * sin(1) = 1.0 * 0.5403023 + 1.0 * 0.84147098 = 1.381773
  // grad_in_odd = -g_even * sin(1) + g_odd * cos(1) = -1.0 * 0.84147098 + 1.0 * 0.5403023 = -0.301168
  std::vector<float> grad_host = input->grad()->to_host();
  EXPECT_NEAR(grad_host[4], 1.381773f, 1e-4f);
  EXPECT_NEAR(grad_host[5], -0.301168f, 1e-4f);
}

TEST(RopeSwigluTest, SwigluNumericalCorrectness) {
  // Test SwiGLU Dense projections
  size_t in_features = 4;
  size_t out_features = 8;
  SwiGLU layer(in_features, out_features);

  std::vector<float> input_data = {0.5f, -0.5f, 1.0f, -1.0f};
  auto input = std::make_shared<Tensor>(Shape{1, in_features}, input_data);
  input->set_requires_grad(true);

  auto output = layer.forward(input);
  EXPECT_EQ(output->shape(), (Shape{1, in_features})); // Down proj returns back to in_features

  auto loss = output->sum({});
  loss->backward();

  // Gradients should propagate back to all internal parameters and inputs
  EXPECT_TRUE(input->grad() != nullptr);
  EXPECT_EQ(input->grad()->shape(), input->shape());

  for (auto &param : layer.parameters()) {
    EXPECT_TRUE(param->grad() != nullptr);
    EXPECT_EQ(param->grad()->shape(), param->shape());
  }
}

TEST(RopeSwigluTest, TransformerDecoderEndToEndWithRope) {
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

  // Autoregressive generation with KV caching and advanced samplers
  auto generated = model.generate_advanced(prompt, max_new_tokens, 0.0f, 0, 1.0f, 42, true);
  EXPECT_EQ(generated.size(), prompt.size() + max_new_tokens);

  // Perform full sequence forward-backward training updates
  model.set_training(true);
  std::vector<float> train_data = {0.0f, 1.0f, 2.0f, 3.0f};
  auto x_train = std::make_shared<Tensor>(Shape{1, 4}, train_data);

  auto logits = model.forward(x_train);
  EXPECT_EQ(logits->shape(), (Shape{1, 4, vocab_size}));

  auto loss = logits->sum({});
  loss->backward();

  for (auto &param : model.parameters()) {
    EXPECT_TRUE(param->grad() != nullptr);
    EXPECT_EQ(param->grad()->shape(), param->shape());
  }
}
