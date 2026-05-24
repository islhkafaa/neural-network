#include "optim/trainer.hpp"

Trainer::Trainer(std::shared_ptr<Layer> model, std::shared_ptr<Loss> criterion,
                 std::shared_ptr<Optimizer> optimizer,
                 std::shared_ptr<LRScheduler> scheduler)
    : model_(model), criterion_(criterion), optimizer_(optimizer),
      scheduler_(scheduler) {}

std::vector<Trainer::EpochHistory>
Trainer::fit(DataLoader &train_loader, int epochs, DataLoader *val_loader) {
  std::vector<EpochHistory> history;
  history.reserve(epochs);

  for (int epoch = 0; epoch < epochs; ++epoch) {
    model_->set_training(true);
    train_loader.reset();

    float total_train_loss = 0.0f;
    size_t train_batches = 0;

    while (train_loader.has_next()) {
      auto batch = train_loader.next();
      auto &x = batch.first;
      auto &y = batch.second;

      optimizer_->zero_grad();
      auto pred = model_->forward(x);
      auto loss = criterion_->forward(pred, y);
      loss->backward();
      optimizer_->step();

      total_train_loss += loss->to_host()[0];
      train_batches++;
    }

    float avg_train_loss = total_train_loss / train_batches;
    float avg_val_loss = 0.0f;

    if (val_loader) {
      avg_val_loss = evaluate(*val_loader);
    }

    if (scheduler_) {
      scheduler_->step();
    }

    history.push_back({avg_train_loss, avg_val_loss});
  }

  return history;
}

float Trainer::evaluate(DataLoader &loader) {
  model_->set_training(false);
  loader.reset();

  float total_loss = 0.0f;
  size_t batches = 0;

  while (loader.has_next()) {
    auto batch = loader.next();
    auto &x = batch.first;
    auto &y = batch.second;

    auto pred = model_->forward(x);
    auto loss = criterion_->forward(pred, y);

    total_loss += loss->to_host()[0];
    batches++;
  }

  return total_loss / batches;
}
