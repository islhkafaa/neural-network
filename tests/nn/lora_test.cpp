#include "nn/layer.hpp"
#include "nn/lora.hpp"
#include <gtest/gtest.h>

TEST(LoRATest, AdapterInitIsNoop) {
  auto base = std::make_shared<Dense>(4, 3);
  LoRAConfig cfg{2, 4.0f, {"weight"}};
  auto lora = std::make_shared<LoRALinear>(base, cfg);

  auto x = std::make_shared<Tensor>(
      Shape{2, 4},
      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});

  auto base_out = base->forward(x);
  auto lora_out = lora->forward(x);

  // Since lora_B is initialized to all zeroes, the output must be identical
  auto base_data = base_out->to_host();
  auto lora_data = lora_out->to_host();

  EXPECT_EQ(base_data.size(), lora_data.size());
  for (size_t i = 0; i < base_data.size(); ++i) {
    EXPECT_FLOAT_EQ(base_data[i], lora_data[i]);
  }
}

TEST(LoRATest, GradFlowsToAdapterOnly) {
  auto base = std::make_shared<Dense>(4, 3);
  LoRAConfig cfg{2, 4.0f, {"weight"}};
  auto lora = std::make_shared<LoRALinear>(base, cfg);

  auto x = std::make_shared<Tensor>(
      Shape{2, 4},
      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  x->set_requires_grad(true);

  // Set non-zero value in lora_B so gradients flow through it
  lora->lora_B()->fill(1.0f);

  auto out = lora->forward(x);
  auto loss = out->sum({});
  loss->backward();

  // Gradients must accumulate in lora_A and lora_B
  EXPECT_TRUE(lora->lora_A()->grad() != nullptr);
  EXPECT_TRUE(lora->lora_B()->grad() != nullptr);

  // Base parameters must be completely frozen and have no gradients
  for (const auto &p : base->parameters()) {
    EXPECT_FALSE(p->requires_grad());
    EXPECT_TRUE(p->grad() == nullptr);
  }
}

TEST(LoRATest, MergeUnmerge) {
  auto base = std::make_shared<Dense>(4, 3);
  LoRAConfig cfg{2, 4.0f, {"weight"}};
  auto lora = std::make_shared<LoRALinear>(base, cfg);

  auto x = std::make_shared<Tensor>(
      Shape{2, 4},
      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});

  // Make the adapter non-zero so merge changes the output
  lora->lora_A()->fill(0.5f);
  lora->lora_B()->fill(0.5f);

  auto pre_merge_out = lora->forward(x)->to_host();

  // Merge the weights
  lora->merge();
  EXPECT_TRUE(lora->is_merged());

  // In merged state, the wrapped base layer's forward pass should match the
  // adapter's pre-merge output
  auto post_merge_out = base->forward(x)->to_host();
  EXPECT_EQ(pre_merge_out.size(), post_merge_out.size());
  for (size_t i = 0; i < pre_merge_out.size(); ++i) {
    EXPECT_NEAR(pre_merge_out[i], post_merge_out[i], 1e-5f);
  }

  // Unmerge the weights
  lora->unmerge();
  EXPECT_FALSE(lora->is_merged());

  // After unmerging, the base layer should produce the original un-adapted
  // outputs again
  lora->lora_B()->fill(0.0f); // Reset adapter path to zero
  auto original_out = base->forward(x)->to_host();
  auto pre_merge_zero_out = lora->forward(x)->to_host();

  EXPECT_EQ(original_out.size(), pre_merge_zero_out.size());
  for (size_t i = 0; i < original_out.size(); ++i) {
    EXPECT_NEAR(original_out[i], pre_merge_zero_out[i], 1e-5f);
  }
}

TEST(LoRATest, MultiHeadAttentionPatch) {
  auto mha = std::make_shared<MultiHeadAttention>(8, 2);

  // In a standard PEFT fine-tuning workflow, we freeze the entire base model
  // first
  for (auto &p : mha->parameters()) {
    p->set_requires_grad(false);
  }

  LoRAConfig cfg{2, 4.0f, {"q_proj", "v_proj"}};

  mha->apply_lora(cfg);
  auto modules = mha->lora_modules();

  // We targeted "q_proj" and "v_proj"
  EXPECT_EQ(modules.size(), 2);

  // Verify that parameter listing inside MultiHeadAttention automatically
  // returns only adapter weights for the patched layers
  auto params = mha->parameters();
  // Unpatched: k_proj (weight, bias) = 2, out_proj (weight, bias) = 2. Total 4.
  // Patched: q_proj (lora_A, lora_B) = 2, v_proj (lora_A, lora_B) = 2. Total 4.
  // Total parameters returned by MHA must be 8
  EXPECT_EQ(params.size(), 8);

  // Verify that base weights are indeed frozen and adapter weights are active
  size_t trainable_count = 0;
  for (const auto &p : params) {
    if (p->requires_grad()) {
      trainable_count++;
    }
  }
  EXPECT_EQ(trainable_count, 4); // Only the 4 adapter parameters (2 for q_proj,
                                 // 2 for v_proj) are trainable
}
