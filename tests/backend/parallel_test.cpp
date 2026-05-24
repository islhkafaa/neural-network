#include "core/tensor.hpp"
#include "backend/thread_pool.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <numeric>
#include <atomic>

TEST(ParallelTest, ThreadPoolCorrectness) {
  size_t size = 5000;
  std::vector<int> data(size, 0);

  // Parallel increment
  ThreadPool::instance().parallel_for(0, size, [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      data[i] = static_cast<int>(i * 2);
    }
  });

  // Verify
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(data[i], static_cast<int>(i * 2));
  }
}

TEST(ParallelTest, ElementwiseThreadSafety) {
  size_t size = 10000;
  std::vector<float> a_data(size, 1.5f);
  std::vector<float> b_data(size, 2.5f);

  auto a = std::make_shared<Tensor>(Shape{size}, a_data);
  auto b = std::make_shared<Tensor>(Shape{size}, b_data);

  // Parallel math triggered internally when elements >= 2048
  auto out = a->add(b);
  std::vector<float> host_out = out->to_host();

  EXPECT_EQ(host_out.size(), size);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_NEAR(host_out[i], 4.0f, 1e-5f);
  }
}

TEST(ParallelTest, AutogradConcurrencySafety) {
  // Verifies that gradient tape backward executes cleanly with multiple operators
  size_t size = 5000;
  std::vector<float> w_data(size, 0.5f);
  std::vector<float> x_data(size, 2.0f);

  auto w = std::make_shared<Tensor>(Shape{size}, w_data);
  auto x = std::make_shared<Tensor>(Shape{size}, x_data);

  w->set_requires_grad(true);
  x->set_requires_grad(true);

  // Out = w * x + x
  auto wx = w->mul(x);
  auto out = wx->add(x);

  auto loss = out->sum({});
  loss->backward();

  // Mathematical gradients:
  // dL/dw = x = 2.0
  // dL/dx = w + 1.0 = 1.5
  std::vector<float> dw = w->grad()->to_host();
  std::vector<float> dx = x->grad()->to_host();

  EXPECT_EQ(dw.size(), size);
  EXPECT_EQ(dx.size(), size);

  for (size_t i = 0; i < size; ++i) {
    EXPECT_NEAR(dw[i], 2.0f, 1e-5f);
    EXPECT_NEAR(dx[i], 1.5f, 1e-5f);
  }
}
