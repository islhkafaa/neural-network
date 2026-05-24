#include "backend/cpu_backend.hpp"
#include "optim/dataloader.hpp"
#include "nn/layer.hpp"
#include "nn/loss.hpp"
#include "optim/lr_scheduler.hpp"
#include "optim/optimizer.hpp"
#include "optim/trainer.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

TEST(PipelineTest, EndToEndLinearRegression) {
  CpuBackend backend;

  // 1. Generate synthetic linear dataset: Y = X * W + B
  // Target weights: w = [2.0, -3.0], bias = 0.5
  size_t train_samples = 80;
  size_t val_samples = 20;
  size_t in_features = 2;
  size_t out_features = 1;

  auto train_x =
      std::make_shared<Tensor>(Shape{train_samples, in_features}, &backend);
  auto train_y =
      std::make_shared<Tensor>(Shape{train_samples, out_features}, &backend);
  auto val_x =
      std::make_shared<Tensor>(Shape{val_samples, in_features}, &backend);
  auto val_y =
      std::make_shared<Tensor>(Shape{val_samples, out_features}, &backend);

  std::vector<float> host_train_x(train_samples * in_features);
  std::vector<float> host_train_y(train_samples * out_features);
  for (size_t i = 0; i < train_samples; ++i) {
    float x1 = static_cast<float>(i) / 100.0f;
    float x2 = static_cast<float>(i % 5) - 2.0f;
    host_train_x[i * in_features + 0] = x1;
    host_train_x[i * in_features + 1] = x2;
    host_train_y[i] = 2.0f * x1 - 3.0f * x2 + 0.5f;
  }
  train_x->copy_from_host(host_train_x);
  train_y->copy_from_host(host_train_y);

  std::vector<float> host_val_x(val_samples * in_features);
  std::vector<float> host_val_y(val_samples * out_features);
  for (size_t i = 0; i < val_samples; ++i) {
    float x1 = static_cast<float>(i + train_samples) / 100.0f;
    float x2 = static_cast<float>((i + train_samples) % 5) - 2.0f;
    host_val_x[i * in_features + 0] = x1;
    host_val_x[i * in_features + 1] = x2;
    host_val_y[i] = 2.0f * x1 - 3.0f * x2 + 0.5f;
  }
  val_x->copy_from_host(host_val_x);
  val_y->copy_from_host(host_val_y);

  // 2. Wrap in Dataset & DataLoader
  auto train_dataset = std::make_shared<TensorDataset>(train_x, train_y);
  auto val_dataset = std::make_shared<TensorDataset>(val_x, val_y);

  DataLoader train_loader(train_dataset, 16, true);
  DataLoader val_loader(val_dataset, 20, false);

  // 3. Construct Model (Sequential containing a single Dense layer)
  auto model = std::make_shared<Sequential>();
  model->add(std::make_shared<Dense>(in_features, out_features, &backend));

  auto criterion = std::make_shared<MSELoss>();

  // 4. Optimizer and LR Scheduler
  float initial_lr = 0.1f;
  auto optimizer = std::make_shared<SGD>(model->parameters(), initial_lr, 0.9f);
  auto scheduler = std::make_shared<StepLR>(*optimizer, 10, 0.5f);

  // 5. Trainer
  Trainer trainer(model, criterion, optimizer, scheduler);

  // Evaluate initial loss
  float initial_loss = trainer.evaluate(val_loader);
  EXPECT_GT(initial_loss, 0.0f);

  // Train for 40 epochs
  auto history = trainer.fit(train_loader, 40, &val_loader);

  // 6. Verify Convergence
  EXPECT_EQ(history.size(), 40);
  float final_loss = history.back().val_loss;

  // The final loss should be significantly smaller than the initial loss, near
  // zero
  EXPECT_LT(final_loss, initial_loss);
  EXPECT_LT(final_loss, 0.05f);
}
