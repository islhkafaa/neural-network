#ifndef QUANTIZATION_HPP
#define QUANTIZATION_HPP

#include "nn/layer.hpp"
#include <memory>
#include <vector>

enum class QuantizationType { INT8, INT4 };

struct QuantizationConfig {
  QuantizationType type = QuantizationType::INT8;
  size_t group_size = 0;
};

class QuantizedDense : public Layer {
public:
  QuantizedDense(std::shared_ptr<Dense> float_dense,
                 const QuantizationConfig &config);

  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
  std::vector<std::shared_ptr<Tensor>> parameters() override;

  std::shared_ptr<Tensor> dequantize_weight() const;

  QuantizationType type() const { return type_; }
  size_t group_size() const { return group_size_; }
  const std::vector<uint8_t> &packed_weights() const { return packed_weights_; }
  std::shared_ptr<Tensor> scales() const { return scales_; }
  std::shared_ptr<Tensor> bias() const { return bias_; }
  Shape weight_shape() const { return weight_shape_; }

private:
  QuantizationType type_;
  size_t group_size_;
  Shape weight_shape_;
  std::vector<uint8_t> packed_weights_;
  std::shared_ptr<Tensor> scales_;
  std::shared_ptr<Tensor> bias_;
  ExecutionBackend *backend_;
};

void quantize_model(Layer *model, const QuantizationConfig &config);

#endif // QUANTIZATION_HPP
