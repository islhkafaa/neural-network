#include "nn/lora.hpp"
#include "core/serialization.hpp"
#include "nn/initialization.hpp"
#include "optim/grad_scaler.hpp"
#include <stdexcept>

LoRALinear::LoRALinear(std::shared_ptr<Dense> base_layer,
                       const LoRAConfig &config)
    : base_layer_(base_layer), rank_(config.rank), alpha_(config.alpha),
      scaling_(config.alpha / static_cast<float>(config.rank)), merged_(false) {
  if (!base_layer_) {
    throw std::runtime_error("LoRALinear requires a valid base Dense layer");
  }

  for (auto &p : base_layer_->parameters()) {
    p->set_requires_grad(false);
  }

  auto base_weight = base_layer_->parameters()[0];
  size_t in_features = base_weight->shape()[0];
  size_t out_features = base_weight->shape()[1];

  lora_A_ = std::make_shared<Tensor>(
      Shape{in_features, rank_}, base_weight->backend(), base_weight->dtype());
  lora_A_->set_requires_grad(true);
  init::kaiming_uniform(*lora_A_);

  lora_B_ = std::make_shared<Tensor>(
      Shape{rank_, out_features}, base_weight->backend(), base_weight->dtype());
  lora_B_->set_requires_grad(true);
  init::constant(*lora_B_, 0.0f);
}

std::shared_ptr<Tensor>
LoRALinear::forward(const std::shared_ptr<Tensor> &input) {
  if (merged_) {
    return base_layer_->forward(input);
  }

  auto base_out = base_layer_->forward(input);

  if (AMPContext::is_enabled()) {
    auto dt = AMPContext::active_dtype();
    auto active_input = input->to(dt);
    auto active_A = lora_A_->to(dt);
    auto active_B = lora_B_->to(dt);

    auto lora_out = active_input->matmul(active_A)->matmul(active_B);
    auto scaled_lora = lora_out * scaling_;
    return base_out + scaled_lora;
  } else {
    auto lora_out = input->matmul(lora_A_)->matmul(lora_B_);
    auto scaled_lora = lora_out * scaling_;
    return base_out + scaled_lora;
  }
}

std::vector<std::shared_ptr<Tensor>> LoRALinear::parameters() {
  if (merged_) {
    return {};
  }
  return {lora_A_, lora_B_};
}

void LoRALinear::set_training(bool training) {
  Layer::set_training(training);
  base_layer_->set_training(training);
}

void LoRALinear::merge() {
  if (merged_)
    return;

  auto delta = lora_A_->matmul(lora_B_) * scaling_;
  auto base_weight = base_layer_->parameters()[0];
  auto new_weight = base_weight + delta;

  base_weight->copy_from_host(new_weight->to_host());
  merged_ = true;
}

void LoRALinear::unmerge() {
  if (!merged_)
    return;

  auto delta = lora_A_->matmul(lora_B_) * scaling_;
  auto base_weight = base_layer_->parameters()[0];
  auto new_weight = base_weight - delta;

  base_weight->copy_from_host(new_weight->to_host());
  merged_ = false;
}

void save_lora_weights(const LoRALinear &layer, const std::string &filepath) {
  serialization::save_parameters({layer.lora_A(), layer.lora_B()}, filepath);
}

void load_lora_weights(LoRALinear &layer, const std::string &filepath) {
  serialization::load_parameters({layer.lora_A(), layer.lora_B()}, filepath);
}
