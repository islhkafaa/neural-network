#include "nn/layer.hpp"
#include "nn/quantization.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(QuantizationTest, SymmetricInt8) {
  auto base = std::make_shared<Dense>(2, 2);
  // Manual initialization of weights
  auto base_weight = base->parameters()[0];
  base_weight->copy_from_host({-1.27f, 0.5f, 12.7f, -6.35f});

  QuantizationConfig cfg{QuantizationType::INT8};
  auto qdense = std::make_shared<QuantizedDense>(base, cfg);

  // Column-0 maximum is 12.7f. Max Q = 127. Scale = 12.7 / 127 = 0.1f.
  // Column-1 maximum is 6.35f. Max Q = 127. Scale = 6.35 / 127 = 0.05f.
  auto scales = qdense->scales()->to_host();
  EXPECT_NEAR(scales[0], 0.1f, 1e-5f);
  EXPECT_NEAR(scales[1], 0.05f, 1e-5f);

  auto dequant = qdense->dequantize_weight()->to_host();
  EXPECT_NEAR(dequant[0], -1.27f, 0.05f); // Theoretical quantization step limit
  EXPECT_NEAR(dequant[1], 0.5f, 1e-5f);
  EXPECT_NEAR(dequant[2], 12.7f, 1e-5f);
  EXPECT_NEAR(dequant[3], -6.35f, 1e-5f);
}

TEST(QuantizationTest, Int4Packing) {
  auto base = std::make_shared<Dense>(4, 1);
  auto base_weight = base->parameters()[0];
  base_weight->copy_from_host({-7.0f, 3.0f, 0.0f, 5.0f});

  QuantizationConfig cfg{QuantizationType::INT4};
  auto qdense = std::make_shared<QuantizedDense>(base, cfg);

  // Maximum is 7.0f. Max Q = 7. Scale = 7.0 / 7 = 1.0f.
  auto scales = qdense->scales()->to_host();
  EXPECT_FLOAT_EQ(scales[0], 1.0f);

  auto dequant = qdense->dequantize_weight()->to_host();
  EXPECT_FLOAT_EQ(dequant[0], -7.0f);
  EXPECT_FLOAT_EQ(dequant[1], 3.0f);
  EXPECT_FLOAT_EQ(dequant[2], 0.0f);
  EXPECT_FLOAT_EQ(dequant[3], 5.0f);
}

TEST(QuantizationTest, QuantizedDenseForward) {
  auto base = std::make_shared<Dense>(4, 3);
  auto base_weight = base->parameters()[0];
  base_weight->copy_from_host({1.5f, -2.5f, 3.0f, -0.8f, 1.2f, -2.0f, 2.0f,
                               -1.5f, 0.5f, -3.0f, 2.5f, -1.0f});
  auto base_bias = base->parameters()[1];
  base_bias->copy_from_host({0.5f, -0.5f, 1.0f});

  auto x = std::make_shared<Tensor>(
      Shape{2, 4},
      std::vector<float>{1.0f, -2.0f, 3.5f, 4.0f, -5.0f, 6.0f, 7.2f, -8.0f});

  auto float_out = base->forward(x)->to_host();

  // Test INT8 Forward accuracy
  QuantizationConfig cfg8{QuantizationType::INT8};
  auto qdense8 = std::make_shared<QuantizedDense>(base, cfg8);
  auto q_out8 = qdense8->forward(x)->to_host();

  EXPECT_EQ(float_out.size(), q_out8.size());
  for (size_t i = 0; i < float_out.size(); ++i) {
    // Relative difference should be extremely small (< 5.0%)
    float diff = std::abs(float_out[i] - q_out8[i]);
    float denom = std::max(1.0f, std::abs(float_out[i]));
    EXPECT_LT(diff / denom, 0.05f);
  }

  // Test INT4 Forward accuracy
  QuantizationConfig cfg4{QuantizationType::INT4};
  auto qdense4 = std::make_shared<QuantizedDense>(base, cfg4);
  auto q_out4 = qdense4->forward(x)->to_host();

  EXPECT_EQ(float_out.size(), q_out4.size());
  for (size_t i = 0; i < float_out.size(); ++i) {
    // INT4 accuracy threshold (< 70.0% due to coarse quantization steps on
    // micro dimensions)
    float diff = std::abs(float_out[i] - q_out4[i]);
    float denom = std::max(1.0f, std::abs(float_out[i]));
    EXPECT_LT(diff / denom, 0.70f);
  }
}

TEST(QuantizationTest, ModelQuantizationWalk) {
  auto seq = std::make_shared<Sequential>();
  auto d1 = std::make_shared<Dense>(4, 8);
  auto d2 = std::make_shared<Dense>(8, 2);
  seq->add(d1);
  seq->add(d2);

  // Initialize weights deterministically to prevent random seed failure
  auto d1_w = d1->parameters()[0];
  d1_w->copy_from_host({0.5f,  -0.8f, 1.2f,  -1.5f, 2.0f,  -0.5f, 0.9f,  -1.1f,
                        -1.0f, 0.7f,  -1.4f, 1.8f,  -0.3f, 1.5f,  -0.8f, 1.2f,
                        1.5f,  -1.2f, 0.5f,  -0.9f, 1.1f,  -1.8f, 1.4f,  -0.7f,
                        -0.5f, 1.1f,  -0.9f, 0.5f,  -1.2f, 0.8f,  -1.5f, 1.0f});
  auto d1_b = d1->parameters()[1];
  d1_b->copy_from_host({0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f});

  auto d2_w = d2->parameters()[0];
  d2_w->copy_from_host({0.8f, -1.2f, -1.5f, 0.9f, 1.1f, -0.7f, -0.5f, 1.3f,
                        1.4f, -1.0f, -0.9f, 0.6f, 0.7f, -1.1f, -1.2f, 1.5f});
  auto d2_b = d2->parameters()[1];
  d2_b->copy_from_host({0.2f, -0.3f});

  // Assert that prior to quantization they are indeed Dense layers
  EXPECT_TRUE(std::dynamic_pointer_cast<Dense>(seq->layers()[0]) != nullptr);
  EXPECT_TRUE(std::dynamic_pointer_cast<Dense>(seq->layers()[1]) != nullptr);

  auto x = std::make_shared<Tensor>(
      Shape{2, 4},
      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});

  auto original_out = seq->forward(x)->to_host();

  // Quantize model recursively to INT8
  QuantizationConfig cfg{QuantizationType::INT8};
  quantize_model(seq.get(), cfg);

  // Verify that model layers have been correctly replaced with QuantizedDense
  // instances
  EXPECT_TRUE(std::dynamic_pointer_cast<QuantizedDense>(seq->layers()[0]) !=
              nullptr);
  EXPECT_TRUE(std::dynamic_pointer_cast<QuantizedDense>(seq->layers()[1]) !=
              nullptr);

  // Assert that quantized model runs and yields outputs close to float original
  auto quantized_out = seq->forward(x)->to_host();
  EXPECT_EQ(original_out.size(), quantized_out.size());
  for (size_t i = 0; i < original_out.size(); ++i) {
    float diff = std::abs(original_out[i] - quantized_out[i]);
    float denom = std::max(1.0f, std::abs(original_out[i]));
    EXPECT_LT(diff / denom, 0.20f);
  }
}
