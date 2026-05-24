#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "optim/dataloader.hpp"
#include "nn/layer.hpp"
#include "nn/loss.hpp"
#include "optim/lr_scheduler.hpp"
#include "optim/optimizer.hpp"
#include <memory>
#include <vector>

class Trainer {
public:
  Trainer(std::shared_ptr<Layer> model, std::shared_ptr<Loss> criterion,
          std::shared_ptr<Optimizer> optimizer,
          std::shared_ptr<LRScheduler> scheduler = nullptr);

  struct EpochHistory {
    float train_loss;
    float val_loss;
  };

  std::vector<EpochHistory> fit(DataLoader &train_loader, int epochs,
                                DataLoader *val_loader = nullptr);

  float evaluate(DataLoader &loader);

private:
  std::shared_ptr<Layer> model_;
  std::shared_ptr<Loss> criterion_;
  std::shared_ptr<Optimizer> optimizer_;
  std::shared_ptr<LRScheduler> scheduler_;
};

#endif // TRAINER_HPP
