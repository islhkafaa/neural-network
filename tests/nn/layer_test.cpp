#include "nn/layer.hpp"
#include "nn/loss.hpp"
#include <gtest/gtest.h>

TEST(LayerTest, DenseForwardBackward) {
  auto dense = std::make_shared<Dense>(3, 2);
  auto x = std::make_shared<Tensor>(
      Shape{2, 3}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  x->set_requires_grad(true);

  auto y = dense->forward(x);
  EXPECT_EQ(y->shape().size(), 2);
  EXPECT_EQ(y->shape()[0], 2);
  EXPECT_EQ(y->shape()[1], 2);

  auto loss = y->sum({});
  loss->backward();

  EXPECT_TRUE(x->grad() != nullptr);
  EXPECT_EQ(x->grad()->shape(), x->shape());
  for (const auto &param : dense->parameters()) {
    EXPECT_TRUE(param->grad() != nullptr);
  }
}

TEST(LayerTest, DropoutEvaluationAndTraining) {
  auto dropout = std::make_shared<Dropout>(0.5f);
  auto x = std::make_shared<Tensor>(
      Shape{1, 10}, std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                       1.0f, 1.0f, 1.0f});

  // Eval mode
  dropout->set_training(false);
  auto y_eval = dropout->forward(x);
  auto eval_data = y_eval->to_host();
  for (float v : eval_data) {
    EXPECT_FLOAT_EQ(v, 1.0f);
  }

  // Training mode
  dropout->set_training(true);
  auto y_train = dropout->forward(x);
  auto train_data = y_train->to_host();
  size_t zeros = 0;
  size_t scaled = 0;
  for (float v : train_data) {
    if (v == 0.0f) {
      zeros++;
    } else if (v == 2.0f) {
      scaled++;
    }
  }
  EXPECT_EQ(zeros + scaled, 10);
}

TEST(LayerTest, LayerNormForward) {
  auto ln = std::make_shared<LayerNorm>(Shape{3});
  auto x = std::make_shared<Tensor>(
      Shape{2, 3}, std::vector<float>{1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f});
  x->set_requires_grad(true);

  auto y = ln->forward(x);
  auto y_data = y->to_host();

  // For each batch element, the mean should be 0.0f and standard deviation
  // should be 1.0f. Batch 0: mean = 2, std = sqrt((1 + 0 + 1) / 3) = sqrt(2/3)
  // = 0.81649658 elements: (1-2)/0.816 = -1.2247, (2-2)/0.816 = 0, (3-2)/0.816
  // = 1.2247
  EXPECT_NEAR(y_data[0], -1.2247f, 1e-3);
  EXPECT_NEAR(y_data[1], 0.0f, 1e-3);
  EXPECT_NEAR(y_data[2], 1.2247f, 1e-3);

  auto loss = y->sum({});
  loss->backward();

  EXPECT_TRUE(x->grad() != nullptr);
  for (const auto &param : ln->parameters()) {
    EXPECT_TRUE(param->grad() != nullptr);
  }
}

TEST(LayerTest, EmbeddingForwardBackward) {
  auto emb = std::make_shared<Embedding>(5, 4);
  auto indices = std::make_shared<Tensor>(
      Shape{2, 2}, std::vector<float>{0.0f, 2.0f, 1.0f, 4.0f});

  auto y = emb->forward(indices);
  EXPECT_EQ(y->shape().size(), 3);
  EXPECT_EQ(y->shape()[0], 2);
  EXPECT_EQ(y->shape()[1], 2);
  EXPECT_EQ(y->shape()[2], 4);

  auto loss = y->sum({});
  loss->backward();

  for (const auto &param : emb->parameters()) {
    EXPECT_TRUE(param->grad() != nullptr);
  }
}

TEST(LayerTest, Conv2DAndMaxPooling2D) {
  auto conv = std::make_shared<Conv2D>(1, 1, 3, 1, 1);
  auto pool = std::make_shared<MaxPooling2D>(2, 2);

  // Input of shape [1, 1, 4, 4]
  auto x = std::make_shared<Tensor>(
      Shape{1, 1, 4, 4},
      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                         10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
  x->set_requires_grad(true);

  auto y_conv = conv->forward(x);
  EXPECT_EQ(y_conv->shape(), Shape({1, 1, 4, 4}));

  auto y_pool = pool->forward(y_conv);
  EXPECT_EQ(y_pool->shape(), Shape({1, 1, 2, 2}));

  auto loss = y_pool->sum({});
  loss->backward();

  EXPECT_TRUE(x->grad() != nullptr);
  for (const auto &param : conv->parameters()) {
    EXPECT_TRUE(param->grad() != nullptr);
  }
}

TEST(LayerTest, RNNSequentialForwardBackward) {
  std::cout << "[RNN Test] Initializing RNN..." << std::endl;
  auto rnn = std::make_shared<RNN>(2, 3);
  // Input: [seq_len=2, batch_size=2, input_size=2]
  std::cout << "[RNN Test] Initializing Input..." << std::endl;
  auto x = std::make_shared<Tensor>(
      Shape{2, 2, 2},
      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  x->set_requires_grad(true);

  std::cout << "[RNN Test] Running forward..." << std::endl;
  auto y = rnn->forward(x);
  EXPECT_EQ(y->shape(), Shape({2, 2, 3}));

  std::cout << "[RNN Test] Computing loss..." << std::endl;
  auto loss = y->sum({});
  std::cout << "[RNN Test] Running backward..." << std::endl;
  loss->backward();

  std::cout << "[RNN Test] Checking gradients..." << std::endl;
  EXPECT_TRUE(x->grad() != nullptr);
  for (const auto &param : rnn->parameters()) {
    EXPECT_TRUE(param->grad() != nullptr);
  }
  std::cout << "[RNN Test] Completed successfully!" << std::endl;
}

TEST(LayerTest, LSTMSequentialForwardBackward) {
  auto lstm = std::make_shared<LSTM>(2, 3);
  // Input: [seq_len=2, batch_size=2, input_size=2]
  auto x = std::make_shared<Tensor>(
      Shape{2, 2, 2},
      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  x->set_requires_grad(true);

  auto y = lstm->forward(x);
  EXPECT_EQ(y->shape(), Shape({2, 2, 3}));

  auto loss = y->sum({});
  loss->backward();

  EXPECT_TRUE(x->grad() != nullptr);
  for (const auto &param : lstm->parameters()) {
    EXPECT_TRUE(param->grad() != nullptr);
  }
}

TEST(LossTest, MSELossEvaluation) {
  auto mse = std::make_shared<MSELoss>();
  auto pred = std::make_shared<Tensor>(
      Shape{2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  auto target = std::make_shared<Tensor>(
      Shape{2, 2}, std::vector<float>{1.5f, 2.5f, 2.5f, 3.5f});
  pred->set_requires_grad(true);

  auto loss = mse->forward(pred, target);
  auto loss_val = loss->to_host()[0];
  // (0.5^2 * 4) / 4 = 0.25f
  EXPECT_FLOAT_EQ(loss_val, 0.25f);

  loss->backward();
  EXPECT_TRUE(pred->grad() != nullptr);
}

TEST(LossTest, BCELossEvaluation) {
  auto bce = std::make_shared<BCELoss>();
  auto pred =
      std::make_shared<Tensor>(Shape{2}, std::vector<float>{0.5f, 0.5f});
  auto target =
      std::make_shared<Tensor>(Shape{2}, std::vector<float>{1.0f, 0.0f});
  pred->set_requires_grad(true);

  auto loss = bce->forward(pred, target);
  auto loss_val = loss->to_host()[0];
  // -1/2 * (1 * log(0.5) + 1 * log(0.5)) = -log(0.5) = 0.69314718
  EXPECT_NEAR(loss_val, 0.69314718f, 1e-5);

  loss->backward();
  EXPECT_TRUE(pred->grad() != nullptr);
}

TEST(LossTest, SoftmaxCrossEntropyLossEvaluation) {
  auto loss_fn = std::make_shared<SoftmaxCrossEntropyLoss>();
  // 2 samples, 3 classes
  auto pred = std::make_shared<Tensor>(
      Shape{2, 3}, std::vector<float>{1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f});
  auto target = std::make_shared<Tensor>(
      Shape{2, 3}, std::vector<float>{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  pred->set_requires_grad(true);

  auto loss = loss_fn->forward(pred, target);
  EXPECT_TRUE(loss->to_host()[0] > 0.0f);

  loss->backward();
  EXPECT_TRUE(pred->grad() != nullptr);
}
