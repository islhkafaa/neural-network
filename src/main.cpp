#include "backend/dml_backend.hpp"
#include "nn/layer.hpp"
#include "nn/loss.hpp"
#include "optim/optimizer.hpp"
#include "core/tensor.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main() {
  std::cout << "=========================================================\n";
  std::cout << "                NEURAL NETWORK ENGINE v2.0               \n";
  std::cout << "=========================================================\n";
  std::cout << "[Device] Initializing computation backend environments...\n";

  [[maybe_unused]] bool has_gpu = false;
  try {
    DmlBackend dml;
    std::cout
        << "[Device] DirectML GPU acceleration successfully initialized.\n";
    has_gpu = true;
  } catch (const std::exception &) {
    std::cout << "[Device] Running on native CPU execution backend.\n";
  }
  std::cout << "=========================================================\n\n";

  std::cout << "--- PART 1: Core Strided Autograd Verification ---\n";

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
  auto y = Wx->add(b);
  auto loss_val = y->sum({0, 1});

  std::cout << "Forward Loss: " << loss_val->to_host()[0]
            << " (Expected: 28.0)\n";

  auto y_val = y->to_host();
  std::cout << "Output vector y: [" << y_val[0] << ", " << y_val[1]
            << "] (Expected: [9.0, 19.0])\n";

  loss_val->backward();

  auto dW = W->grad()->to_host();
  std::cout << "Gradient dW: [" << dW[0] << ", " << dW[1] << ", " << dW[2]
            << "; " << dW[3] << ", " << dW[4] << ", " << dW[5] << "]\n"
            << "             (Expected: [0.5, 1.0, 1.5; 0.5, 1.0, 1.5])\n";

  auto dx = x->grad()->to_host();
  std::cout << "Gradient dx: [" << dx[0] << ", " << dx[1] << ", " << dx[2]
            << "]\n"
            << "             (Expected: [5.0, 7.0, 9.0])\n";

  auto db = b->grad()->to_host();
  std::cout << "Gradient db: [" << db[0] << ", " << db[1] << "]\n"
            << "             (Expected: [1.0, 1.0])\n\n";

  std::cout << "--- PART 2: TinyGPT with Rotary Position Embeddings (RoPE) & "
               "SwiGLU ---\n";

  const std::vector<char> vocab = {'n', 'e', 't', 'w', 'o', 'r', 'k'};
  const size_t vocab_size = vocab.size();
  const size_t embed_dim = 16;
  const size_t num_heads = 2;
  const size_t dim_feedforward = 32;
  const size_t num_layers = 1;
  const size_t max_seq_len = 8;

  std::cout << "Vocabulary: ";
  for (size_t i = 0; i < vocab.size(); ++i) {
    std::cout << "'" << vocab[i] << "' (" << i << ")  ";
  }
  std::cout << "\nTarget word to memorize: \"network\"\n\n";

  TransformerDecoder model(vocab_size, embed_dim, num_heads, dim_feedforward,
                           num_layers, max_seq_len);

  std::cout << "Generating text with UNTRAINED model (greedy):\n";
  std::vector<size_t> prompt = {0};
  auto gen_untrained =
      model.generate_advanced(prompt, 6, 0.0f, 0, 1.0f, 42, true);

  std::cout << "  Prompt: \"" << vocab[prompt[0]] << "\"\n";
  std::cout << "  Generated output IDs: ";
  for (size_t id : gen_untrained)
    std::cout << id << " ";
  std::cout << "\n  Generated output text: \"";
  for (size_t id : gen_untrained)
    std::cout << vocab[id];
  std::cout << "\"\n\n";

  const size_t seq_len = 6;
  std::vector<float> input_ids = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  auto x_train = std::make_shared<Tensor>(Shape{1, seq_len}, input_ids);

  std::vector<float> target_one_hot(1 * seq_len * vocab_size, 0.0f);
  std::vector<size_t> target_ids = {1, 2, 3, 4, 5, 6};
  for (size_t s = 0; s < seq_len; ++s) {
    size_t target_tok = target_ids[s];
    target_one_hot[s * vocab_size + target_tok] = 1.0f;
  }
  auto y_train =
      std::make_shared<Tensor>(Shape{1, seq_len, vocab_size}, target_one_hot);

  AdamW optimizer(model.parameters(), 0.015f, 0.9f, 0.999f, 1e-8f, 0.01f);
  SoftmaxCrossEntropyLoss criterion;

  std::cout << "Training the model to memorize \"network\" over 60 epochs:\n";
  model.set_training(true);

  for (size_t epoch = 1; epoch <= 60; ++epoch) {
    optimizer.zero_grad();
    auto logits = model.forward(x_train);
    auto loss = criterion.forward(logits, y_train);
    loss->backward();
    optimizer.step();

    if (epoch % 10 == 0 || epoch == 1) {
      std::cout << "  Epoch " << epoch << "/60 | Loss: " << loss->to_host()[0]
                << "\n";
    }
  }
  std::cout << "Training completed.\n\n";

  model.set_training(false);
  std::cout << "Generating text with TRAINED model using KV Caching (greedy, "
               "temp=0):\n";
  auto gen_greedy = model.generate_advanced(prompt, 6, 0.0f, 0, 1.0f, 42, true);

  std::cout << "  Generated output text: \"";
  for (size_t id : gen_greedy)
    std::cout << vocab[id];
  std::cout << "\"\n\n";

  std::cout << "Generating text with TRAINED model using KV Caching & Nucleus "
               "Sampler (temp=1.0, top-k=3, top-p=0.9):\n";
  auto gen_sampled =
      model.generate_advanced(prompt, 6, 1.00f, 3, 0.9f, 99, true);

  std::cout << "  Generated output text: \"";
  for (size_t id : gen_sampled)
    std::cout << vocab[id];
  std::cout << "\"\n";
  std::cout << "=========================================================\n";

  return 0;
}
