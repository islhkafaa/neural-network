#include "optim/optimizer.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

Optimizer::Optimizer(std::vector<std::shared_ptr<Tensor>> params, float lr,
                     float weight_decay)
    : params_(std::move(params)), lr_(lr), weight_decay_(weight_decay) {}

void Optimizer::zero_grad() noexcept {
  for (auto &p : params_) {
    if (p) {
      p->zero_grad();
    }
  }
}

void Optimizer::clip_grad_norm(float max_norm) {
  if (max_norm <= 0.0f) {
    throw std::runtime_error("clip_grad_norm: max_norm must be positive.");
  }
  float total_norm_sq = 0.0f;
  for (auto &p : params_) {
    if (p && p->grad()) {
      auto g_data = p->grad()->to_host();
      for (float val : g_data) {
        total_norm_sq += val * val;
      }
    }
  }
  float total_norm = std::sqrt(total_norm_sq);
  if (total_norm > max_norm) {
    float clip_coef = max_norm / (total_norm + 1e-6f);
    for (auto &p : params_) {
      if (p && p->grad()) {
        auto g_data = p->grad()->to_host();
        for (float &val : g_data) {
          val *= clip_coef;
        }
        p->grad()->copy_from_host(g_data);
      }
    }
  }
}

void Optimizer::clip_grad_value(float clip_value) {
  if (clip_value <= 0.0f) {
    throw std::runtime_error("clip_grad_value: clip_value must be positive.");
  }
  for (auto &p : params_) {
    if (p && p->grad()) {
      auto g_data = p->grad()->to_host();
      for (float &val : g_data) {
        val = std::clamp(val, -clip_value, clip_value);
      }
      p->grad()->copy_from_host(g_data);
    }
  }
}

SGD::SGD(std::vector<std::shared_ptr<Tensor>> params, float lr, float momentum,
         float weight_decay, bool nesterov)
    : Optimizer(std::move(params), lr, weight_decay), momentum_(momentum),
      nesterov_(nesterov) {
  velocities_.resize(params_.size());
  for (size_t i = 0; i < params_.size(); ++i) {
    if (params_[i]) {
      velocities_[i].assign(count_elements(params_[i]->shape()), 0.0f);
    }
  }
}

void SGD::step() {
  for (size_t i = 0; i < params_.size(); ++i) {
    auto &p = params_[i];
    if (!p || !p->grad())
      continue;

    auto p_data = p->to_host();
    auto g_data = p->grad()->to_host();
    size_t size = p_data.size();
    auto &v = velocities_[i];

    for (size_t j = 0; j < size; ++j) {
      float grad_val = g_data[j] + weight_decay_ * p_data[j];
      if (momentum_ > 0.0f) {
        v[j] = momentum_ * v[j] + grad_val;
        if (nesterov_) {
          grad_val = grad_val + momentum_ * v[j];
        } else {
          grad_val = v[j];
        }
      }
      p_data[j] -= lr_ * grad_val;
    }
    p->copy_from_host(p_data);
  }
}

Adam::Adam(std::vector<std::shared_ptr<Tensor>> params, float lr, float beta1,
           float beta2, float eps, float weight_decay)
    : Optimizer(std::move(params), lr, weight_decay), beta1_(beta1),
      beta2_(beta2), eps_(eps) {
  m_.resize(params_.size());
  v_.resize(params_.size());
  for (size_t i = 0; i < params_.size(); ++i) {
    if (params_[i]) {
      size_t num_el = count_elements(params_[i]->shape());
      m_[i].assign(num_el, 0.0f);
      v_[i].assign(num_el, 0.0f);
    }
  }
}

void Adam::step() {
  t_++;
  float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
  float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

  for (size_t i = 0; i < params_.size(); ++i) {
    auto &p = params_[i];
    if (!p || !p->grad())
      continue;

    auto p_data = p->to_host();
    auto g_data = p->grad()->to_host();
    size_t size = p_data.size();
    auto &m_vec = m_[i];
    auto &v_vec = v_[i];

    for (size_t j = 0; j < size; ++j) {
      float grad_val = g_data[j] + weight_decay_ * p_data[j];
      m_vec[j] = beta1_ * m_vec[j] + (1.0f - beta1_) * grad_val;
      v_vec[j] = beta2_ * v_vec[j] + (1.0f - beta2_) * grad_val * grad_val;

      float m_hat = m_vec[j] / bias_correction1;
      float v_hat = v_vec[j] / bias_correction2;

      p_data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
    }
    p->copy_from_host(p_data);
  }
}

RMSprop::RMSprop(std::vector<std::shared_ptr<Tensor>> params, float lr,
                 float alpha, float eps, float weight_decay, float momentum)
    : Optimizer(std::move(params), lr, weight_decay), alpha_(alpha), eps_(eps),
      momentum_(momentum) {
  square_avgs_.resize(params_.size());
  velocities_.resize(params_.size());
  for (size_t i = 0; i < params_.size(); ++i) {
    if (params_[i]) {
      size_t num_el = count_elements(params_[i]->shape());
      square_avgs_[i].assign(num_el, 0.0f);
      velocities_[i].assign(num_el, 0.0f);
    }
  }
}

void RMSprop::step() {
  for (size_t i = 0; i < params_.size(); ++i) {
    auto &p = params_[i];
    if (!p || !p->grad())
      continue;

    auto p_data = p->to_host();
    auto g_data = p->grad()->to_host();
    size_t size = p_data.size();
    auto &sq_avg = square_avgs_[i];
    auto &v = velocities_[i];

    for (size_t j = 0; j < size; ++j) {
      float grad_val = g_data[j] + weight_decay_ * p_data[j];
      sq_avg[j] = alpha_ * sq_avg[j] + (1.0f - alpha_) * grad_val * grad_val;

      float step_val = grad_val / (std::sqrt(sq_avg[j]) + eps_);
      if (momentum_ > 0.0f) {
        v[j] = momentum_ * v[j] + step_val;
        p_data[j] -= lr_ * v[j];
      } else {
        p_data[j] -= lr_ * step_val;
      }
    }
    p->copy_from_host(p_data);
  }
}

AdamW::AdamW(std::vector<std::shared_ptr<Tensor>> params, float lr, float beta1,
             float beta2, float eps, float weight_decay)
    : Optimizer(std::move(params), lr, weight_decay), beta1_(beta1),
      beta2_(beta2), eps_(eps) {
  m_.resize(params_.size());
  v_.resize(params_.size());
  for (size_t i = 0; i < params_.size(); ++i) {
    if (params_[i]) {
      size_t num_el = count_elements(params_[i]->shape());
      m_[i].assign(num_el, 0.0f);
      v_[i].assign(num_el, 0.0f);
    }
  }
}

void AdamW::step() {
  t_++;
  float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
  float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

  for (size_t i = 0; i < params_.size(); ++i) {
    auto &p = params_[i];
    if (!p || !p->grad())
      continue;

    auto p_data = p->to_host();
    auto g_data = p->grad()->to_host();
    size_t size = p_data.size();
    auto &m_vec = m_[i];
    auto &v_vec = v_[i];

    for (size_t j = 0; j < size; ++j) {
      float grad_val = g_data[j];
      m_vec[j] = beta1_ * m_vec[j] + (1.0f - beta1_) * grad_val;
      v_vec[j] = beta2_ * v_vec[j] + (1.0f - beta2_) * grad_val * grad_val;

      float m_hat = m_vec[j] / bias_correction1;
      float v_hat = v_vec[j] / bias_correction2;

      p_data[j] = p_data[j] - lr_ * weight_decay_ * p_data[j];
      p_data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
    }
    p->copy_from_host(p_data);
  }
}

Adafactor::Adafactor(std::vector<std::shared_ptr<Tensor>> params, float lr,
                     float beta2, float eps1, float eps2, float weight_decay)
    : Optimizer(std::move(params), lr, weight_decay), beta2_(beta2),
      eps1_(eps1), eps2_(eps2) {
  r_.resize(params_.size());
  c_.resize(params_.size());
  v_.resize(params_.size());
  for (size_t i = 0; i < params_.size(); ++i) {
    if (params_[i]) {
      const auto &sh = params_[i]->shape();
      if (sh.size() == 2) {
        r_[i].assign(sh[0], 0.0f);
        c_[i].assign(sh[1], 0.0f);
      } else {
        v_[i].assign(count_elements(sh), 0.0f);
      }
    }
  }
}

void Adafactor::step() {
  for (size_t i = 0; i < params_.size(); ++i) {
    auto &p = params_[i];
    if (!p || !p->grad())
      continue;

    auto p_data = p->to_host();
    auto g_data = p->grad()->to_host();
    size_t size = p_data.size();
    const auto &sh = p->shape();

    if (sh.size() == 2) {
      size_t d1 = sh[0];
      size_t d2 = sh[1];
      auto &r_vec = r_[i];
      auto &c_vec = c_[i];

      std::vector<float> row_sums(d1, 0.0f);
      std::vector<float> col_sums(d2, 0.0f);
      for (size_t r = 0; r < d1; ++r) {
        for (size_t col = 0; col < d2; ++col) {
          float g2 = g_data[r * d2 + col] * g_data[r * d2 + col] + eps1_;
          row_sums[r] += g2;
          col_sums[col] += g2;
        }
      }

      float r_sum_total = 0.0f;
      for (size_t r = 0; r < d1; ++r) {
        r_vec[r] = beta2_ * r_vec[r] + (1.0f - beta2_) * row_sums[r];
        r_sum_total += r_vec[r];
      }
      for (size_t col = 0; col < d2; ++col) {
        c_vec[col] = beta2_ * c_vec[col] + (1.0f - beta2_) * col_sums[col];
      }

      for (size_t r = 0; r < d1; ++r) {
        for (size_t col = 0; col < d2; ++col) {
          size_t idx = r * d2 + col;
          float v_val = (r_vec[r] * c_vec[col]) / (r_sum_total + 1e-12f);
          float denom = std::sqrt(v_val) + eps2_;

          p_data[idx] = p_data[idx] - lr_ * weight_decay_ * p_data[idx];
          p_data[idx] -= lr_ * g_data[idx] / denom;
        }
      }
    } else {
      auto &v_vec = v_[i];
      for (size_t j = 0; j < size; ++j) {
        float g2 = g_data[j] * g_data[j] + eps1_;
        v_vec[j] = beta2_ * v_vec[j] + (1.0f - beta2_) * g2;
        float denom = std::sqrt(v_vec[j]) + eps2_;

        p_data[j] = p_data[j] - lr_ * weight_decay_ * p_data[j];
        p_data[j] -= lr_ * g_data[j] / denom;
      }
    }
    p->copy_from_host(p_data);
  }
}
