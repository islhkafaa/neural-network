#ifndef LORA_HPP
#define LORA_HPP

#include "nn/layer.hpp"
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

struct LoRAConfig {
  size_t rank = 4;
  float alpha = 8.0f;
  std::unordered_set<std::string> target_modules;
};

class LoRALinear : public Layer {
public:
  LoRALinear(std::shared_ptr<Dense> base_layer, const LoRAConfig &config);

  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
  std::vector<std::shared_ptr<Tensor>> parameters() override;
  void set_training(bool training) override;

  void merge();
  void unmerge();

  std::shared_ptr<Dense> base_layer() const { return base_layer_; }
  std::shared_ptr<Tensor> lora_A() const { return lora_A_; }
  std::shared_ptr<Tensor> lora_B() const { return lora_B_; }
  size_t rank() const { return rank_; }
  float alpha() const { return alpha_; }
  bool is_merged() const { return merged_; }

private:
  std::shared_ptr<Dense> base_layer_;
  size_t rank_;
  float alpha_;
  float scaling_;
  std::shared_ptr<Tensor> lora_A_;
  std::shared_ptr<Tensor> lora_B_;
  bool merged_ = false;
};

void save_lora_weights(const LoRALinear &layer, const std::string &filepath);
void load_lora_weights(LoRALinear &layer, const std::string &filepath);

#endif // LORA_HPP
