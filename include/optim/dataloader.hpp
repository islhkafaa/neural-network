#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include "core/tensor.hpp"
#include <memory>
#include <utility>
#include <vector>

class Dataset {
public:
  virtual ~Dataset() = default;
  virtual size_t size() const = 0;
  virtual std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
  get_item(size_t index) = 0;
};

class TensorDataset : public Dataset {
public:
  TensorDataset(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y);
  size_t size() const override;
  std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
  get_item(size_t index) override;

private:
  std::shared_ptr<Tensor> x_;
  std::shared_ptr<Tensor> y_;
};

class DataLoader {
public:
  DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size,
             bool shuffle = true);

  void reset();
  bool has_next() const;
  std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> next();

  size_t batch_size() const { return batch_size_; }
  size_t num_batches() const;

private:
  std::shared_ptr<Dataset> dataset_;
  size_t batch_size_;
  bool shuffle_;
  size_t current_index_;
  std::vector<size_t> indices_;
};

#endif // DATALOADER_HPP
