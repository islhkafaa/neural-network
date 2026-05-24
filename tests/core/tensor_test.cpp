#include "core/shape.hpp"
#include "core/tensor.hpp"
#include <gtest/gtest.h>

TEST(ShapeTest, CountElements) {
  EXPECT_EQ(count_elements(Shape{}), 1);
  EXPECT_EQ(count_elements(Shape{2, 3}), 6);
  EXPECT_EQ(count_elements(Shape{4, 1, 5}), 20);
}

TEST(ShapeTest, ComputeStrides) {
  Shape s{2, 3, 4};
  Strides strides = compute_strides(s);
  ASSERT_EQ(strides.size(), 3);
  EXPECT_EQ(strides[0], 12);
  EXPECT_EQ(strides[1], 4);
  EXPECT_EQ(strides[2], 1);
}

TEST(ShapeTest, IsContiguous) {
  Shape s{2, 3};
  Strides strides = compute_strides(s);
  EXPECT_TRUE(is_contiguous(s, strides));

  Strides non_contiguous_strides{1, 2};
  EXPECT_FALSE(is_contiguous(s, non_contiguous_strides));
}

TEST(TensorTest, BasicConstruction) {
  auto t = std::make_shared<Tensor>(Shape{2, 3});
  t->fill(5.5f);
  auto host = t->to_host();
  ASSERT_EQ(host.size(), 6);
  for (float val : host) {
    EXPECT_FLOAT_EQ(val, 5.5f);
  }
}

TEST(TensorTest, ElementwiseAdd) {
  auto a = std::make_shared<Tensor>(Shape{2, 2},
                                    std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  auto b = std::make_shared<Tensor>(Shape{2, 2},
                                    std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f});
  auto c = a + b;
  auto res = c->to_host();
  ASSERT_EQ(res.size(), 4);
  EXPECT_FLOAT_EQ(res[0], 6.0f);
  EXPECT_FLOAT_EQ(res[1], 8.0f);
  EXPECT_FLOAT_EQ(res[2], 10.0f);
  EXPECT_FLOAT_EQ(res[3], 12.0f);
}

TEST(TensorTest, BroadcastingAdd) {
  auto a = std::make_shared<Tensor>(
      Shape{2, 3}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto b = std::make_shared<Tensor>(Shape{1, 3},
                                    std::vector<float>{10.0f, 20.0f, 30.0f});
  auto c = a + b;
  auto res = c->to_host();
  ASSERT_EQ(res.size(), 6);
  EXPECT_FLOAT_EQ(res[0], 11.0f);
  EXPECT_FLOAT_EQ(res[1], 22.0f);
  EXPECT_FLOAT_EQ(res[2], 33.0f);
  EXPECT_FLOAT_EQ(res[3], 14.0f);
  EXPECT_FLOAT_EQ(res[4], 25.0f);
  EXPECT_FLOAT_EQ(res[5], 36.0f);
}

TEST(TensorTest, ElementwiseSubAndMul) {
  auto a = std::make_shared<Tensor>(Shape{2}, std::vector<float>{10.0f, 20.0f});
  auto b = std::make_shared<Tensor>(Shape{2}, std::vector<float>{2.0f, 5.0f});

  auto sub_res = (a - b)->to_host();
  EXPECT_FLOAT_EQ(sub_res[0], 8.0f);
  EXPECT_FLOAT_EQ(sub_res[1], 15.0f);

  auto mul_res = (a * b)->to_host();
  EXPECT_FLOAT_EQ(mul_res[0], 20.0f);
  EXPECT_FLOAT_EQ(mul_res[1], 100.0f);
}

TEST(TensorTest, MatmulContiguous) {
  auto a = std::make_shared<Tensor>(
      Shape{2, 3}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto b = std::make_shared<Tensor>(
      Shape{3, 2}, std::vector<float>{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  auto c = a->matmul(b);
  auto res = c->to_host();
  ASSERT_EQ(res.size(), 4);
  EXPECT_FLOAT_EQ(res[0], 58.0f);
  EXPECT_FLOAT_EQ(res[1], 64.0f);
  EXPECT_FLOAT_EQ(res[2], 139.0f);
  EXPECT_FLOAT_EQ(res[3], 154.0f);
}

TEST(TensorTest, SumReduction) {
  auto t = std::make_shared<Tensor>(
      Shape{2, 3}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  auto s_all = t->sum({0, 1}, false);
  EXPECT_EQ(s_all->shape().size(), 0);
  EXPECT_FLOAT_EQ(s_all->to_host()[0], 21.0f);

  auto s_col = t->sum({0}, false);
  auto s_col_val = s_col->to_host();
  ASSERT_EQ(s_col_val.size(), 3);
  EXPECT_FLOAT_EQ(s_col_val[0], 5.0f);
  EXPECT_FLOAT_EQ(s_col_val[1], 7.0f);
  EXPECT_FLOAT_EQ(s_col_val[2], 9.0f);

  auto s_row = t->sum({1}, true);
  ASSERT_EQ(s_row->shape().size(), 2);
  EXPECT_EQ(s_row->shape()[0], 2);
  EXPECT_EQ(s_row->shape()[1], 1);
  auto s_row_val = s_row->to_host();
  EXPECT_FLOAT_EQ(s_row_val[0], 6.0f);
  EXPECT_FLOAT_EQ(s_row_val[1], 15.0f);
}

TEST(AutogradTest, SimpleScalarGraph) {
  auto a = std::make_shared<Tensor>(Shape{}, std::vector<float>{2.0f});
  auto b = std::make_shared<Tensor>(Shape{}, std::vector<float>{3.0f});
  auto c = std::make_shared<Tensor>(Shape{}, std::vector<float>{4.0f});

  a->set_requires_grad(true);
  b->set_requires_grad(true);
  c->set_requires_grad(true);

  auto ab = a + b;
  auto y = ab * c;

  y->backward();

  EXPECT_FLOAT_EQ(y->to_host()[0], 20.0f);
  EXPECT_FLOAT_EQ(a->grad()->to_host()[0], 4.0f);
  EXPECT_FLOAT_EQ(b->grad()->to_host()[0], 4.0f);
  EXPECT_FLOAT_EQ(c->grad()->to_host()[0], 5.0f);
}

TEST(AutogradTest, ComplexDagWithBroadcasting) {
  auto W = std::make_shared<Tensor>(
      Shape{2, 3}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto x = std::make_shared<Tensor>(Shape{3, 1},
                                    std::vector<float>{0.5f, 1.0f, 1.5f});
  auto b =
      std::make_shared<Tensor>(Shape{2, 1}, std::vector<float>{2.0f, 3.0f});

  W->set_requires_grad(true);
  x->set_requires_grad(true);
  b->set_requires_grad(true);

  auto Wx = W->matmul(x);
  auto y = Wx + b;
  auto loss = y->sum({});

  loss->backward();

  auto dW = W->grad()->to_host();
  EXPECT_FLOAT_EQ(dW[0], 0.5f);
  EXPECT_FLOAT_EQ(dW[1], 1.0f);
  EXPECT_FLOAT_EQ(dW[2], 1.5f);
  EXPECT_FLOAT_EQ(dW[3], 0.5f);
  EXPECT_FLOAT_EQ(dW[4], 1.0f);
  EXPECT_FLOAT_EQ(dW[5], 1.5f);

  auto dx = x->grad()->to_host();
  EXPECT_FLOAT_EQ(dx[0], 5.0f);
  EXPECT_FLOAT_EQ(dx[1], 7.0f);
  EXPECT_FLOAT_EQ(dx[2], 9.0f);

  auto db = b->grad()->to_host();
  EXPECT_FLOAT_EQ(db[0], 1.0f);
  EXPECT_FLOAT_EQ(db[1], 1.0f);
}
