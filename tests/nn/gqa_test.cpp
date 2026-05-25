#include "nn/layer.hpp"
#include <gtest/gtest.h>

TEST(GQATest, DimensionVerification) {
  size_t embed_dim = 16;
  size_t num_heads = 4;
  size_t num_kv_heads = 2; // GQA
  size_t B = 2;
  size_t S = 4;

  auto mha = std::make_shared<MultiHeadAttention>(embed_dim, num_heads, false,
                                                  nullptr, num_kv_heads);
  auto input = std::make_shared<Tensor>(Shape{B, S, embed_dim});
  std::vector<float> input_data(B * S * embed_dim, 0.1f);
  input->copy_from_host(input_data);

  auto output = mha->forward(input);
  EXPECT_EQ(output->shape().size(), 3);
  EXPECT_EQ(output->shape()[0], B);
  EXPECT_EQ(output->shape()[1], S);
  EXPECT_EQ(output->shape()[2], embed_dim);
}

TEST(GQATest, MQADimensionVerification) {
  size_t embed_dim = 16;
  size_t num_heads = 4;
  size_t num_kv_heads = 1; // MQA
  size_t B = 2;
  size_t S = 4;

  auto mha = std::make_shared<MultiHeadAttention>(embed_dim, num_heads, false,
                                                  nullptr, num_kv_heads);
  auto input = std::make_shared<Tensor>(Shape{B, S, embed_dim});
  std::vector<float> input_data(B * S * embed_dim, 0.2f);
  input->copy_from_host(input_data);

  auto output = mha->forward(input);
  EXPECT_EQ(output->shape().size(), 3);
  EXPECT_EQ(output->shape()[0], B);
  EXPECT_EQ(output->shape()[1], S);
  EXPECT_EQ(output->shape()[2], embed_dim);
}

TEST(GQATest, EquivalentToMHAWhenHeadsEqual) {
  size_t embed_dim = 8;
  size_t num_heads = 2;
  size_t B = 1;
  size_t S = 3;

  auto mha_std = std::make_shared<MultiHeadAttention>(embed_dim, num_heads,
                                                      false, nullptr, 0);
  auto mha_gqa = std::make_shared<MultiHeadAttention>(
      embed_dim, num_heads, false, nullptr, num_heads);

  // Copy identical parameters
  auto std_params = mha_std->parameters();
  auto gqa_params = mha_gqa->parameters();
  ASSERT_EQ(std_params.size(), gqa_params.size());

  for (size_t i = 0; i < std_params.size(); ++i) {
    gqa_params[i]->copy_from_host(std_params[i]->to_host());
  }

  auto input = std::make_shared<Tensor>(Shape{B, S, embed_dim});
  std::vector<float> input_data = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f,
                                   0.7f, -0.8f, 0.9f, 1.0f, -1.1f, 1.2f,
                                   1.3f, -1.4f, 1.5f, 1.6f, -1.7f, 1.8f,
                                   1.9f, -2.0f, 2.1f, 2.2f, -2.3f, 2.4f};
  input->copy_from_host(input_data);

  auto out_std = mha_std->forward(input);
  auto out_gqa = mha_gqa->forward(input);

  auto std_data = out_std->to_host();
  auto gqa_data = out_gqa->to_host();

  ASSERT_EQ(std_data.size(), gqa_data.size());
  for (size_t i = 0; i < std_data.size(); ++i) {
    EXPECT_NEAR(std_data[i], gqa_data[i], 1e-5f);
  }
}

TEST(GQATest, KVCacheSupport) {
  size_t embed_dim = 8;
  size_t num_heads = 4;
  size_t num_kv_heads = 2;
  size_t B = 1;

  auto mha = std::make_shared<MultiHeadAttention>(embed_dim, num_heads, false,
                                                  nullptr, num_kv_heads);

  KVCache cache;

  // Step 1: Forward first token
  auto input1 = std::make_shared<Tensor>(Shape{B, 1, embed_dim});
  input1->copy_from_host({0.1f, 0.2f, -0.3f, 0.4f, 0.5f, -0.6f, 0.7f, 0.8f});
  auto out1 = mha->forward(input1, &cache);

  ASSERT_NE(cache.k, nullptr);
  ASSERT_NE(cache.v, nullptr);
  // Shape should be (B, num_kv_heads, S, head_dim)
  EXPECT_EQ(cache.k->shape()[0], B);
  EXPECT_EQ(cache.k->shape()[1], num_kv_heads);
  EXPECT_EQ(cache.k->shape()[2], 1);
  EXPECT_EQ(cache.k->shape()[3], embed_dim / num_heads);

  // Step 2: Forward second token
  auto input2 = std::make_shared<Tensor>(Shape{B, 1, embed_dim});
  input2->copy_from_host({-0.8f, 0.7f, 0.6f, -0.5f, 0.4f, 0.3f, -0.2f, 0.1f});
  auto out2 = mha->forward(input2, &cache);

  EXPECT_EQ(cache.k->shape()[2], 2); // S should now be 2
}

TEST(GQATest, FusedAttentionEquivalency) {
  size_t embed_dim = 16;
  size_t num_heads = 4;
  size_t num_kv_heads = 2; // GQA
  size_t B = 2;
  size_t S = 3;

  auto mha = std::make_shared<MultiHeadAttention>(embed_dim, num_heads, true,
                                                  nullptr, num_kv_heads);

  auto input = std::make_shared<Tensor>(Shape{B, S, embed_dim});
  std::vector<float> input_data(B * S * embed_dim);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i) * 0.1f - 0.5f;
  }
  input->copy_from_host(input_data);

  // 1. Run unfused path by enabling gradients on inputs/parameters
  input->set_requires_grad(true);
  for (auto &p : mha->parameters()) {
    p->set_requires_grad(true);
  }
  auto out_unfused = mha->forward(input);
  auto unfused_data = out_unfused->to_host();

  // 2. Run fused path by disabling gradients
  input->set_requires_grad(false);
  for (auto &p : mha->parameters()) {
    p->set_requires_grad(false);
  }
  auto out_fused = mha->forward(input);
  auto fused_data = out_fused->to_host();

  ASSERT_EQ(unfused_data.size(), fused_data.size());
  for (size_t i = 0; i < unfused_data.size(); ++i) {
    EXPECT_NEAR(unfused_data[i], fused_data[i], 1e-4f);
  }
}

TEST(GQATest, FusedAttentionWithKVCache) {
  size_t embed_dim = 8;
  size_t num_heads = 4;
  size_t num_kv_heads = 2;
  size_t B = 1;

  auto mha_unfused = std::make_shared<MultiHeadAttention>(
      embed_dim, num_heads, true, nullptr, num_kv_heads);
  auto mha_fused = std::make_shared<MultiHeadAttention>(
      embed_dim, num_heads, true, nullptr, num_kv_heads);

  auto unfused_params = mha_unfused->parameters();
  auto fused_params = mha_fused->parameters();
  for (size_t i = 0; i < unfused_params.size(); ++i) {
    fused_params[i]->copy_from_host(unfused_params[i]->to_host());
  }

  KVCache cache_unfused;
  KVCache cache_fused;

  // Step 1: Forward first token
  auto input1 = std::make_shared<Tensor>(Shape{B, 1, embed_dim});
  input1->copy_from_host({0.1f, 0.2f, -0.3f, 0.4f, 0.5f, -0.6f, 0.7f, 0.8f});

  input1->set_requires_grad(true);
  for (auto &p : mha_unfused->parameters()) {
    p->set_requires_grad(true);
  }
  auto out1_unfused = mha_unfused->forward(input1, &cache_unfused);

  input1->set_requires_grad(false);
  for (auto &p : mha_fused->parameters()) {
    p->set_requires_grad(false);
  }
  auto out1_fused = mha_fused->forward(input1, &cache_fused);

  auto out1_unfused_data = out1_unfused->to_host();
  auto out1_fused_data = out1_fused->to_host();
  for (size_t i = 0; i < out1_unfused_data.size(); ++i) {
    EXPECT_NEAR(out1_unfused_data[i], out1_fused_data[i], 1e-4f);
  }

  // Step 2: Forward second token
  auto input2 = std::make_shared<Tensor>(Shape{B, 1, embed_dim});
  input2->copy_from_host({-0.8f, 0.7f, 0.6f, -0.5f, 0.4f, 0.3f, -0.2f, 0.1f});

  input2->set_requires_grad(true);
  auto out2_unfused = mha_unfused->forward(input2, &cache_unfused);

  input2->set_requires_grad(false);
  auto out2_fused = mha_fused->forward(input2, &cache_fused);

  auto out2_unfused_data = out2_unfused->to_host();
  auto out2_fused_data = out2_fused->to_host();
  for (size_t i = 0; i < out2_unfused_data.size(); ++i) {
    EXPECT_NEAR(out2_unfused_data[i], out2_fused_data[i], 1e-4f);
  }
}

TEST(GQATest, KVQuantizationVerification) {
  size_t embed_dim = 8;
  size_t num_heads = 4;
  size_t num_kv_heads = 2;
  size_t B = 1;

  auto mha_float = std::make_shared<MultiHeadAttention>(
      embed_dim, num_heads, true, nullptr, num_kv_heads);
  auto mha_quant = std::make_shared<MultiHeadAttention>(
      embed_dim, num_heads, true, nullptr, num_kv_heads);

  // Copy parameters
  auto float_params = mha_float->parameters();
  auto quant_params = mha_quant->parameters();
  for (size_t i = 0; i < float_params.size(); ++i) {
    quant_params[i]->copy_from_host(float_params[i]->to_host());
  }

  KVCache cache_float(false);
  KVCache cache_quant(true);

  // Forward first token
  auto input1 = std::make_shared<Tensor>(Shape{B, 1, embed_dim});
  input1->copy_from_host({0.1f, 0.2f, -0.3f, 0.4f, 0.5f, -0.6f, 0.7f, 0.8f});

  input1->set_requires_grad(false);
  for (auto &p : mha_float->parameters())
    p->set_requires_grad(false);
  for (auto &p : mha_quant->parameters())
    p->set_requires_grad(false);

  auto out1_float = mha_float->forward(input1, &cache_float);
  auto out1_quant = mha_quant->forward(input1, &cache_quant);

  auto out1_float_data = out1_float->to_host();
  auto out1_quant_data = out1_quant->to_host();
  ASSERT_EQ(out1_float_data.size(), out1_quant_data.size());
  for (size_t i = 0; i < out1_float_data.size(); ++i) {
    EXPECT_NEAR(out1_float_data[i], out1_quant_data[i], 0.08f);
  }

  // Forward second token
  auto input2 = std::make_shared<Tensor>(Shape{B, 1, embed_dim});
  input2->copy_from_host({-0.8f, 0.7f, 0.6f, -0.5f, 0.4f, 0.3f, -0.2f, 0.1f});

  auto out2_float = mha_float->forward(input2, &cache_float);
  auto out2_quant = mha_quant->forward(input2, &cache_quant);

  auto out2_float_data = out2_float->to_host();
  auto out2_quant_data = out2_quant->to_host();
  ASSERT_EQ(out2_float_data.size(), out2_quant_data.size());
  for (size_t i = 0; i < out2_float_data.size(); ++i) {
    EXPECT_NEAR(out2_float_data[i], out2_quant_data[i], 0.08f);
  }
}

TEST(GQATest, KVQuantizationBackwardParity) {
  size_t embed_dim = 8;
  size_t num_heads = 4;
  size_t num_kv_heads = 2;
  size_t B = 1;

  auto mha_float = std::make_shared<MultiHeadAttention>(
      embed_dim, num_heads, true, nullptr, num_kv_heads);
  auto mha_quant = std::make_shared<MultiHeadAttention>(
      embed_dim, num_heads, true, nullptr, num_kv_heads);

  // Copy parameters
  auto float_params = mha_float->parameters();
  auto quant_params = mha_quant->parameters();
  for (size_t i = 0; i < float_params.size(); ++i) {
    quant_params[i]->copy_from_host(float_params[i]->to_host());
  }

  KVCache cache_float(false);
  KVCache cache_quant(true);

  // Forward and backward for float
  auto input_float = std::make_shared<Tensor>(Shape{B, 1, embed_dim});
  input_float->copy_from_host(
      {0.1f, 0.2f, -0.3f, 0.4f, 0.5f, -0.6f, 0.7f, 0.8f});
  input_float->set_requires_grad(true);
  for (auto &p : mha_float->parameters())
    p->set_requires_grad(true);

  auto out1_float = mha_float->forward(input_float, &cache_float);
  auto loss_float = out1_float->sum({});
  loss_float->backward();

  // Forward and backward for quant
  auto input_quant = std::make_shared<Tensor>(Shape{B, 1, embed_dim});
  input_quant->copy_from_host(
      {0.1f, 0.2f, -0.3f, 0.4f, 0.5f, -0.6f, 0.7f, 0.8f});
  input_quant->set_requires_grad(true);
  for (auto &p : mha_quant->parameters())
    p->set_requires_grad(true);

  auto out1_quant = mha_quant->forward(input_quant, &cache_quant);
  auto loss_quant = out1_quant->sum({});
  loss_quant->backward();

  // Verify outputs
  auto out1_float_data = out1_float->to_host();
  auto out1_quant_data = out1_quant->to_host();
  ASSERT_EQ(out1_float_data.size(), out1_quant_data.size());
  for (size_t i = 0; i < out1_float_data.size(); ++i) {
    EXPECT_NEAR(out1_float_data[i], out1_quant_data[i], 0.08f);
  }

  // Verify gradients of parameters
  for (size_t i = 0; i < float_params.size(); ++i) {
    if (float_params[i]->grad()) {
      auto float_grad = float_params[i]->grad()->to_host();
      auto quant_grad = quant_params[i]->grad()->to_host();
      ASSERT_EQ(float_grad.size(), quant_grad.size());
      for (size_t j = 0; j < float_grad.size(); ++j) {
        EXPECT_NEAR(float_grad[j], quant_grad[j], 0.08f);
      }
    }
  }
}
