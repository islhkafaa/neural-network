#include "optim/lr_scheduler.hpp"
#include "optim/optimizer.hpp"
#include <gtest/gtest.h>

TEST(OptimizerTest, SGDSimpleStep) {
  auto param = std::make_shared<Tensor>(Shape{1}, std::vector<float>{10.0f});
  param->set_requires_grad(true);

  // Set dummy gradient
  auto grad = std::make_shared<Tensor>(Shape{1}, std::vector<float>{2.0f});
  param->accumulate_grad(grad);

  SGD opt({param}, 0.1f);
  opt.step();

  auto val = param->to_host()[0];
  EXPECT_NEAR(val, 9.8f, 1e-5f);
}

TEST(OptimizerTest, SGDMomentum) {
  auto param = std::make_shared<Tensor>(Shape{1}, std::vector<float>{10.0f});
  param->set_requires_grad(true);

  SGD opt({param}, 0.1f, 0.9f); // lr = 0.1, momentum = 0.9

  // Step 1
  auto grad1 = std::make_shared<Tensor>(Shape{1}, std::vector<float>{2.0f});
  param->accumulate_grad(grad1);
  opt.step();
  EXPECT_NEAR(param->to_host()[0], 9.8f, 1e-5f);

  // Step 2
  opt.zero_grad();
  auto grad2 = std::make_shared<Tensor>(Shape{1}, std::vector<float>{2.0f});
  param->accumulate_grad(grad2);
  opt.step();
  // v_1 = 2.0
  // v_2 = 0.9 * v_1 + 2.0 = 3.8
  // p_2 = 9.8 - 0.1 * 3.8 = 9.42
  EXPECT_NEAR(param->to_host()[0], 9.42f, 1e-5f);
}

TEST(OptimizerTest, AdamSimpleStep) {
  auto param = std::make_shared<Tensor>(Shape{1}, std::vector<float>{1.0f});
  param->set_requires_grad(true);

  auto grad = std::make_shared<Tensor>(Shape{1}, std::vector<float>{0.5f});
  param->accumulate_grad(grad);

  Adam opt({param}, 1e-3f);
  opt.step();

  // m_1 = 0.1 * 0.5 = 0.05
  // v_1 = 0.001 * 0.25 = 0.00025
  // m_hat = 0.05 / 0.1 = 0.5
  // v_hat = 0.00025 / 0.001 = 0.25
  // p_1 = 1.0 - 0.001 * 0.5 / sqrt(0.25) = 1.0 - 0.001 * 0.5 / 0.5 = 0.999f
  EXPECT_NEAR(param->to_host()[0], 0.999f, 1e-5f);
}

TEST(OptimizerTest, RMSpropSimpleStep) {
  auto param = std::make_shared<Tensor>(Shape{1}, std::vector<float>{1.0f});
  param->set_requires_grad(true);

  auto grad = std::make_shared<Tensor>(Shape{1}, std::vector<float>{0.5f});
  param->accumulate_grad(grad);

  RMSprop opt({param}, 1e-2f, 0.99f);
  opt.step();

  // sq_avg_1 = 0.01 * 0.25 = 0.0025
  // p_1 = 1.0 - 0.01 * 0.5 / sqrt(0.0025) = 1.0 - 0.01 * 0.5 / 0.05 = 0.90f
  EXPECT_NEAR(param->to_host()[0], 0.90f, 1e-5f);
}

TEST(OptimizerTest, GradientClippingNorm) {
  auto param1 = std::make_shared<Tensor>(Shape{1}, std::vector<float>{1.0f});
  param1->set_requires_grad(true);
  auto grad1 = std::make_shared<Tensor>(Shape{1}, std::vector<float>{3.0f});
  param1->accumulate_grad(grad1);

  auto param2 = std::make_shared<Tensor>(Shape{1}, std::vector<float>{2.0f});
  param2->set_requires_grad(true);
  auto grad2 = std::make_shared<Tensor>(Shape{1}, std::vector<float>{4.0f});
  param2->accumulate_grad(grad2);

  SGD opt({param1, param2}, 0.1f);

  // grad norm = sqrt(3^2 + 4^2) = 5.0
  // clip at 2.5
  opt.clip_grad_norm(2.5f);

  // Expected scaling factor = 2.5 / 5.0 = 0.5
  // scaled grad1 = 1.5, scaled grad2 = 2.0
  EXPECT_NEAR(param1->grad()->to_host()[0], 1.5f, 1e-5f);
  EXPECT_NEAR(param2->grad()->to_host()[0], 2.0f, 1e-5f);
}

TEST(OptimizerTest, GradientClippingValue) {
  auto param =
      std::make_shared<Tensor>(Shape{2}, std::vector<float>{1.0f, 2.0f});
  param->set_requires_grad(true);
  auto grad =
      std::make_shared<Tensor>(Shape{2}, std::vector<float>{-5.0f, 10.0f});
  param->accumulate_grad(grad);

  SGD opt({param}, 0.1f);
  opt.clip_grad_value(3.0f);

  auto clipped = param->grad()->to_host();
  EXPECT_NEAR(clipped[0], -3.0f, 1e-5f);
  EXPECT_NEAR(clipped[1], 3.0f, 1e-5f);
}

TEST(OptimizerTest, StepLRScheduling) {
  auto param = std::make_shared<Tensor>(Shape{1}, std::vector<float>{1.0f});
  SGD opt({param}, 0.1f);

  StepLR scheduler(opt, 2, 0.5f); // decay by 0.5 every 2 steps

  // Epoch 1
  scheduler.step();
  EXPECT_NEAR(opt.lr(), 0.1f, 1e-5f);

  // Epoch 2
  scheduler.step();
  EXPECT_NEAR(opt.lr(), 0.05f, 1e-5f);

  // Epoch 3
  scheduler.step();
  EXPECT_NEAR(opt.lr(), 0.05f, 1e-5f);

  // Epoch 4
  scheduler.step();
  EXPECT_NEAR(opt.lr(), 0.025f, 1e-5f);
}

TEST(OptimizerTest, CosineAnnealingLRScheduling) {
  auto param = std::make_shared<Tensor>(Shape{1}, std::vector<float>{1.0f});
  SGD opt({param}, 0.1f);

  CosineAnnealingLR scheduler(opt, 10, 0.01f); // T_max = 10, eta_min = 0.01

  // step 1: T_cur = 1
  // lr = 0.01 + 0.5 * 0.09 * (1 + cos(1/10 * pi))
  // cos(pi/10) = cos(0.314159) = 0.9510565
  // lr = 0.01 + 0.045 * 1.9510565 = 0.0977975
  scheduler.step();
  EXPECT_NEAR(opt.lr(), 0.0977975f, 1e-5f);

  // Step 10: T_cur = 10
  // cos(pi) = -1
  // lr = 0.01 + 0.5 * 0.09 * (1 - 1) = 0.01
  for (int i = 0; i < 9; ++i) {
    scheduler.step();
  }
  EXPECT_NEAR(opt.lr(), 0.01f, 1e-5f);
}
