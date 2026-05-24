#include "nn/layer.hpp"
#include "core/tensor.hpp"
#include <cstdio>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

TEST(SerializationTest, SaveAndLoadConsistency) {
  auto backend = default_backend();

  // 1. Create first model and initialize
  auto model = std::make_shared<Sequential>();
  model->add(std::make_shared<Dense>(2, 4, backend));
  model->add(std::make_shared<ReLU>());
  model->add(std::make_shared<Dense>(4, 1, backend));

  // Generate input
  auto input = std::make_shared<Tensor>(Shape{3, 2},
                                        std::vector<float>{
                                            0.5f, -1.0f, // Sample 1
                                            2.0f, 1.5f,  // Sample 2
                                            -0.5f, 0.0f  // Sample 3
                                        },
                                        backend);

  // Compute reference forward pass
  auto ref_output = model->forward(input);
  auto ref_data = ref_output->to_host();

  // Save weights
  const std::string filename = "test_weights.bin";
  model->save_weights(filename);

  // 2. Create second model with identical architecture
  auto loaded_model = std::make_shared<Sequential>();
  loaded_model->add(std::make_shared<Dense>(2, 4, backend));
  loaded_model->add(std::make_shared<ReLU>());
  loaded_model->add(std::make_shared<Dense>(4, 1, backend));

  // Load saved weights
  loaded_model->load_weights(filename);

  // Verify parameters are identical
  auto original_params = model->parameters();
  auto loaded_params = loaded_model->parameters();

  ASSERT_EQ(original_params.size(), loaded_params.size());
  for (size_t i = 0; i < original_params.size(); ++i) {
    auto orig_data = original_params[i]->to_host();
    auto load_data = loaded_params[i]->to_host();
    ASSERT_EQ(orig_data.size(), load_data.size());
    for (size_t j = 0; j < orig_data.size(); ++j) {
      EXPECT_NEAR(orig_data[j], load_data[j], 1e-6f);
    }
  }

  // Compute forward pass with loaded model
  auto loaded_output = loaded_model->forward(input);
  auto loaded_data = loaded_output->to_host();

  // Verify outputs are identical
  ASSERT_EQ(ref_data.size(), loaded_data.size());
  for (size_t i = 0; i < ref_data.size(); ++i) {
    EXPECT_NEAR(ref_data[i], loaded_data[i], 1e-5f);
  }

  // Clean up file
  std::remove(filename.c_str());
}

TEST(SerializationTest, ShapeMismatchAssertion) {
  auto backend = default_backend();

  auto model1 = std::make_shared<Sequential>();
  model1->add(std::make_shared<Dense>(2, 4, backend));

  const std::string filename = "test_shape_mismatch.bin";
  model1->save_weights(filename);

  // Trying to load (2, 4) weights into a (2, 5) model should fail
  auto model2 = std::make_shared<Sequential>();
  model2->add(std::make_shared<Dense>(2, 5, backend));

  EXPECT_THROW(model2->load_weights(filename), std::runtime_error);

  std::remove(filename.c_str());
}
