#include "nn/loss.hpp"

std::shared_ptr<Tensor>
MSELoss::forward(const std::shared_ptr<Tensor> &pred,
                 const std::shared_ptr<Tensor> &target) {
  auto diff = pred->sub(target);
  auto sq_diff = diff->mul(diff);

  std::vector<size_t> axes;
  for (size_t i = 0; i < pred->shape().size(); ++i) {
    axes.push_back(i);
  }
  auto sum_val = sq_diff->sum(axes);

  float N = static_cast<float>(count_elements(pred->shape()));
  auto N_tensor =
      std::make_shared<Tensor>(Shape{}, std::vector<float>{N}, pred->backend());

  return sum_val->div(N_tensor);
}

std::shared_ptr<Tensor>
BCELoss::forward(const std::shared_ptr<Tensor> &pred,
                 const std::shared_ptr<Tensor> &target) {
  auto log_pred = pred->log();
  auto term1 = target->mul(log_pred);

  auto one = std::make_shared<Tensor>(pred->shape(), pred->backend());
  one->fill(1.0f);
  auto one_minus_target = one->sub(target);
  auto one_minus_pred = one->sub(pred);
  auto log_one_minus_pred = one_minus_pred->log();
  auto term2 = one_minus_target->mul(log_one_minus_pred);

  auto sum_terms = term1->add(term2);

  std::vector<size_t> axes;
  for (size_t i = 0; i < pred->shape().size(); ++i) {
    axes.push_back(i);
  }
  auto total_sum = sum_terms->sum(axes);

  float N = static_cast<float>(count_elements(pred->shape()));
  auto negative_inv_N = std::make_shared<Tensor>(
      Shape{}, std::vector<float>{-1.0f / N}, pred->backend());
  return total_sum->mul(negative_inv_N);
}

std::shared_ptr<Tensor>
SoftmaxCrossEntropyLoss::forward(const std::shared_ptr<Tensor> &pred,
                                 const std::shared_ptr<Tensor> &target) {
  auto probs = pred->softmax(pred->shape().size() - 1);
  auto log_probs = probs->log();
  auto term = target->mul(log_probs);

  std::vector<size_t> axes;
  for (size_t i = 0; i < pred->shape().size(); ++i) {
    axes.push_back(i);
  }
  auto total_sum = term->sum(axes);

  float B = static_cast<float>(pred->shape()[0]);
  auto negative_inv_B = std::make_shared<Tensor>(
      Shape{}, std::vector<float>{-1.0f / B}, pred->backend());
  return total_sum->mul(negative_inv_B);
}
