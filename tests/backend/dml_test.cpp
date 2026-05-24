#include "backend/dml_backend.hpp"
#include "core/tensor.hpp"
#include <gtest/gtest.h>

#ifdef WITH_DML
TEST(DmlTest, MemoryTransferAndBasicMath) {
  DmlBackend backend;

  auto buf = backend.allocate(10 * sizeof(float));
  ASSERT_NE(buf, nullptr);

  std::vector<float> host_in = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                                5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  backend.copy_host_to_device(host_in.data(), *buf, 10 * sizeof(float));

  backend.fill(*buf, 5.0f);
  std::vector<float> host_out(10);
  backend.copy_device_to_host(*buf, host_out.data(), 10 * sizeof(float));
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_NEAR(host_out[i], 5.0f, 1e-5f);
  }
}

TEST(DmlTest, MatmulGemm) {
  DmlBackend backend;

  auto a_buf = backend.allocate(6 * sizeof(float));
  auto b_buf = backend.allocate(6 * sizeof(float));
  auto out_buf = backend.allocate(4 * sizeof(float));

  std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> b_data = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  backend.copy_host_to_device(a_data.data(), *a_buf, 6 * sizeof(float));
  backend.copy_host_to_device(b_data.data(), *b_buf, 6 * sizeof(float));

  backend.matmul(*a_buf, 0, Shape{2, 3}, Strides{3, 1}, *b_buf, 0, Shape{3, 2},
                 Strides{2, 1}, *out_buf, 0, Shape{2, 2}, Strides{2, 1});

  std::vector<float> out_data(4);
  backend.copy_device_to_host(*out_buf, out_data.data(), 4 * sizeof(float));

  EXPECT_NEAR(out_data[0], 58.0f, 1e-4f);
  EXPECT_NEAR(out_data[1], 64.0f, 1e-4f);
  EXPECT_NEAR(out_data[2], 139.0f, 1e-4f);
  EXPECT_NEAR(out_data[3], 154.0f, 1e-4f);
}
#else
TEST(DmlTest, DisabledFallback) {
  EXPECT_THROW(
      {
        DmlBackend backend;
        (void)backend.allocate(100);
      },
      std::runtime_error);
}
#endif
