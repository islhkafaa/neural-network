#include <gtest/gtest.h>
#include "core/tensor.hpp"
#include "core/dtype.hpp"
#include "nn/layer.hpp"
#include "optim/optimizer.hpp"
#include "optim/grad_scaler.hpp"
#include <cmath>
#include <vector>
#include <iostream>

TEST(AMPTest, BitConversionsCorrectness) {
  std::cout << "[Trace] BitConversionsCorrectness starting..." << std::endl;
  std::vector<float> values = {0.0f, 1.0f, -2.0f, 0.5f, 3.14159f};
  for (float v : values) {
    uint16_t h = float_to_fp16(v);
    float r = fp16_to_float(h);
    EXPECT_NEAR(v, r, 1e-3f);
  }

  for (float v : values) {
    uint16_t b = float_to_bfloat16(v);
    float r = bfloat16_to_float(b);
    EXPECT_NEAR(v, r, 5e-2f);
  }
  std::cout << "[Trace] BitConversionsCorrectness finished successfully." << std::endl;
}

TEST(AMPTest, LayerMixedPrecision) {
  std::cout << "[Trace] LayerMixedPrecision starting..." << std::endl;
  AMPGuard guard(true, DataType::FP16);

  std::cout << "[Trace] Creating input tensor..." << std::endl;
  auto input = std::make_shared<Tensor>(Shape{2, 3}, std::vector<float>{
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f
  });
  input->set_requires_grad(true);

  std::cout << "[Trace] Creating Dense layer..." << std::endl;
  Dense layer(3, 2);
  EXPECT_EQ(layer.parameters()[0]->dtype(), DataType::FP32);

  std::cout << "[Trace] Running Dense forward pass..." << std::endl;
  auto output = layer.forward(input);
  EXPECT_EQ(output->dtype(), DataType::FP16);

  std::cout << "[Trace] Creating target tensor..." << std::endl;
  auto target = std::make_shared<Tensor>(Shape{2, 2}, std::vector<float>{0.5f, 0.5f, 0.5f, 0.5f});
  
  std::cout << "[Trace] Computing diff = output - target..." << std::endl;
  auto diff = output->sub(target);
  
  std::cout << "[Trace] Computing sq_diff = diff * diff..." << std::endl;
  auto sq_diff = diff->mul(diff);
  
  std::cout << "[Trace] Computing loss = sum(sq_diff)..." << std::endl;
  auto loss = sq_diff->sum({});
  
  std::cout << "[Trace] Triggering backward pass loss->backward()..." << std::endl;
  loss->backward();

  std::cout << "[Trace] Verifying gradients..." << std::endl;
  EXPECT_TRUE(input->grad() != nullptr);
  EXPECT_TRUE(layer.parameters()[0]->grad() != nullptr);

  EXPECT_EQ(layer.parameters()[0]->grad()->dtype(), DataType::FP32);
  EXPECT_EQ(input->grad()->dtype(), DataType::FP32);
  std::cout << "[Trace] LayerMixedPrecision finished successfully." << std::endl;
}

TEST(AMPTest, GradScalerScalingAndUnscaling) {
  std::cout << "[Trace] GradScalerScalingAndUnscaling starting..." << std::endl;
  AMPGuard guard(true, DataType::FP16);

  auto input = std::make_shared<Tensor>(Shape{2, 3}, std::vector<float>{
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f
  });
  Dense layer(3, 2);

  std::vector<std::shared_ptr<Tensor>> params = layer.parameters();
  SGD optimizer(params, 1e-2f);

  GradScaler scaler(1024.0f, 2.0f, 0.5f, 2);

  std::cout << "[Trace] GradScaler Step 1..." << std::endl;
  auto output = layer.forward(input);
  auto loss = output->sum({});
  auto scaled_loss = scaler.scale(loss);
  scaled_loss->backward();

  float initial_grad_sum = 0.0f;
  for (float val : layer.parameters()[0]->grad()->to_host()) {
    initial_grad_sum += std::abs(val);
  }

  scaler.unscale(optimizer);

  float unscaled_grad_sum = 0.0f;
  for (float val : layer.parameters()[0]->grad()->to_host()) {
    unscaled_grad_sum += std::abs(val);
  }

  EXPECT_NEAR(unscaled_grad_sum * 1024.0f, initial_grad_sum, 1e-1f);

  bool step_ok = scaler.step(optimizer);
  EXPECT_TRUE(step_ok);

  scaler.update();
  EXPECT_FLOAT_EQ(scaler.scale_factor(), 1024.0f);

  std::cout << "[Trace] GradScaler Step 2..." << std::endl;
  optimizer.zero_grad();
  output = layer.forward(input);
  loss = output->sum({});
  scaled_loss = scaler.scale(loss);
  scaled_loss->backward();

  scaler.step(optimizer);
  scaler.update();
  EXPECT_FLOAT_EQ(scaler.scale_factor(), 2048.0f);

  std::cout << "[Trace] GradScaler Step 3 (overflow)..." << std::endl;
  optimizer.zero_grad();
  output = layer.forward(input);
  loss = output->sum({});
  scaled_loss = scaler.scale(loss);
  scaled_loss->backward();

  std::vector<float> bad_grad = layer.parameters()[0]->grad()->to_host();
  bad_grad[0] = std::numeric_limits<float>::quiet_NaN();
  layer.parameters()[0]->grad()->copy_from_host(bad_grad);

  bool bad_step_ok = scaler.step(optimizer);
  EXPECT_FALSE(bad_step_ok);

  scaler.update();
  EXPECT_FLOAT_EQ(scaler.scale_factor(), 1024.0f);
  std::cout << "[Trace] GradScalerScalingAndUnscaling finished successfully." << std::endl;
}
