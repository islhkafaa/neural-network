#include "optim/optimizer.hpp"
#include "nn/layer.hpp"
#include <gtest/gtest.h>

TEST(OptimizerAdvancedTest, AdamWDecoupledWeightDecay) {
  // Initialize parameter to 1.0f
  auto param = std::make_shared<Tensor>(Shape{1}, std::vector<float>{1.0f});
  param->set_requires_grad(true);

  // Set dummy gradient to 0.0f (so Adam term is 0)
  auto grad = std::make_shared<Tensor>(Shape{1}, std::vector<float>{0.0f});
  param->accumulate_grad(grad);

  // AdamW: lr = 0.1, weight_decay = 0.5
  AdamW opt({param}, 0.1f, 0.9f, 0.999f, 1e-8f, 0.5f);
  opt.step();

  // Decoupled weight decay: p = 1.0 - lr * weight_decay * p = 1.0 - 0.1 * 0.5 * 1.0 = 0.95
  float val = param->to_host()[0];
  EXPECT_NEAR(val, 0.95f, 1e-5f);
}

TEST(OptimizerAdvancedTest, AdafactorLowRankFactoring) {
  // Initialize a 2D parameter of shape (2, 2)
  std::vector<float> p_data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto param = std::make_shared<Tensor>(Shape{2, 2}, p_data);
  param->set_requires_grad(true);

  // Set dummy gradients
  std::vector<float> g_data = {0.1f, 0.2f, 0.3f, 0.4f};
  auto grad = std::make_shared<Tensor>(Shape{2, 2}, g_data);
  param->accumulate_grad(grad);

  // Run Adafactor with no weight decay
  Adafactor opt({param}, 0.01f, 0.99f, 1e-30f, 1e-3f, 0.0f);
  opt.step();

  auto val = param->to_host();
  EXPECT_EQ(val.size(), 4);
  // Verify updates took place (parameters decreased due to positive gradient)
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_LT(val[i], p_data[i]);
  }
}

TEST(OptimizerAdvancedTest, StochasticDepthGradientFlow) {
  // 1. Drop probability = 1.0f (Always drop)
  {
    auto x = std::make_shared<Tensor>(Shape{2}, std::vector<float>{1.0f, 2.0f});
    auto sub = std::make_shared<Tensor>(Shape{2}, std::vector<float>{3.0f, 4.0f});
    x->set_requires_grad(true);
    sub->set_requires_grad(true);

    StochasticDepth sd(1.0f);
    sd.set_training(true);

    auto out = sd.forward_residual(x, sub);
    EXPECT_NEAR(out->to_host()[0], 1.0f, 1e-5f);
    EXPECT_NEAR(out->to_host()[1], 2.0f, 1e-5f);

    auto loss = out->sum({});
    loss->backward();

    // Gradients should flow only to x, sub should receive no gradient flow
    ASSERT_TRUE(x->grad() != nullptr);
    EXPECT_NEAR(x->grad()->to_host()[0], 1.0f, 1e-5f);
    EXPECT_NEAR(x->grad()->to_host()[1], 1.0f, 1e-5f);
    EXPECT_TRUE(sub->grad() == nullptr);
  }

  // 2. Drop probability = 0.0f (Always keep)
  {
    auto x = std::make_shared<Tensor>(Shape{2}, std::vector<float>{1.0f, 2.0f});
    auto sub = std::make_shared<Tensor>(Shape{2}, std::vector<float>{3.0f, 4.0f});
    x->set_requires_grad(true);
    sub->set_requires_grad(true);

    StochasticDepth sd(0.0f);
    sd.set_training(true);

    auto out = sd.forward_residual(x, sub);
    EXPECT_NEAR(out->to_host()[0], 4.0f, 1e-5f);
    EXPECT_NEAR(out->to_host()[1], 6.0f, 1e-5f);

    auto loss = out->sum({});
    loss->backward();

    // Gradients should flow to both
    ASSERT_TRUE(x->grad() != nullptr);
    ASSERT_TRUE(sub->grad() != nullptr);
    EXPECT_NEAR(x->grad()->to_host()[0], 1.0f, 1e-5f);
    EXPECT_NEAR(sub->grad()->to_host()[0], 1.0f, 1e-5f);
  }

  // 3. Evaluation Mode (Always keep, no scaling)
  {
    auto x = std::make_shared<Tensor>(Shape{2}, std::vector<float>{1.0f, 2.0f});
    auto sub = std::make_shared<Tensor>(Shape{2}, std::vector<float>{3.0f, 4.0f});
    x->set_requires_grad(true);
    sub->set_requires_grad(true);

    StochasticDepth sd(0.5f);
    sd.set_training(false); // EVAL mode

    auto out = sd.forward_residual(x, sub);
    EXPECT_NEAR(out->to_host()[0], 4.0f, 1e-5f);
    EXPECT_NEAR(out->to_host()[1], 6.0f, 1e-5f);

    auto loss = out->sum({});
    loss->backward();

    ASSERT_TRUE(x->grad() != nullptr);
    ASSERT_TRUE(sub->grad() != nullptr);
    EXPECT_NEAR(x->grad()->to_host()[0], 1.0f, 1e-5f);
    EXPECT_NEAR(sub->grad()->to_host()[0], 1.0f, 1e-5f);
  }
}
