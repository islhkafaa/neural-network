#include "optim/dataloader.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

TensorDataset::TensorDataset(std::shared_ptr<Tensor> x,
                             std::shared_ptr<Tensor> y)
    : x_(x), y_(y) {
  if (x->shape().empty() || y->shape().empty()) {
    throw std::runtime_error("TensorDataset: inputs must not be empty/scalar");
  }
  if (x->shape()[0] != y->shape()[0]) {
    throw std::runtime_error(
        "TensorDataset: size mismatch along batch dimension");
  }
}

size_t TensorDataset::size() const { return x_->shape()[0]; }

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
TensorDataset::get_item(size_t index) {
  if (index >= size()) {
    throw std::out_of_range("TensorDataset::get_item: index out of range");
  }
  return {x_->slice(0, index), y_->slice(0, index)};
}

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size,
                       bool shuffle)
    : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle),
      current_index_(0) {
  indices_.resize(dataset_->size());
  std::iota(indices_.begin(), indices_.end(), size_t{0});
  reset();
}

void DataLoader::reset() {
  current_index_ = 0;
  if (shuffle_ && !indices_.empty()) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices_.begin(), indices_.end(), g);
  }
}

bool DataLoader::has_next() const { return current_index_ < indices_.size(); }

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> DataLoader::next() {
  if (!has_next()) {
    throw std::runtime_error("DataLoader: no more batches available");
  }

  size_t actual_batch_size =
      std::min(batch_size_, indices_.size() - current_index_);
  std::vector<std::shared_ptr<Tensor>> x_batch;
  std::vector<std::shared_ptr<Tensor>> y_batch;
  x_batch.reserve(actual_batch_size);
  y_batch.reserve(actual_batch_size);

  for (size_t i = 0; i < actual_batch_size; ++i) {
    size_t idx = indices_[current_index_ + i];
    auto item = dataset_->get_item(idx);
    x_batch.push_back(item.first);
    y_batch.push_back(item.second);
  }

  current_index_ += actual_batch_size;
  return {Tensor::stack(x_batch), Tensor::stack(y_batch)};
}

size_t DataLoader::num_batches() const {
  return (indices_.size() + batch_size_ - 1) / batch_size_;
}
