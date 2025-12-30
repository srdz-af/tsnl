#include "optim.hpp"

#include <cmath>

Parameter::Parameter(size_t n, float init) {
    data.assign(n, init);
    grad.assign(n, 0.0f);
}

void Parameter::zero_grad() {
    std::fill(grad.begin(), grad.end(), 0.0f);
}

Optimizer::Optimizer(const OptimConfig& cfg, const std::vector<Parameter*>& params)
    : cfg_(cfg), params_(params) {
    if (cfg_.use_adam) {
        m_.resize(params_.size());
        v_.resize(params_.size());
        for (size_t i = 0; i < params_.size(); ++i) {
            m_[i].assign(params_[i]->size(), 0.0f);
            v_[i].assign(params_[i]->size(), 0.0f);
        }
    }
}

void Optimizer::zero_grad() {
    for (auto* p : params_) p->zero_grad();
}

void Optimizer::step() {
    ++t_;
    if (cfg_.use_adam) {
        float beta1t = std::pow(cfg_.beta1, static_cast<float>(t_));
        float beta2t = std::pow(cfg_.beta2, static_cast<float>(t_));
        for (size_t pi = 0; pi < params_.size(); ++pi) {
            auto* p = params_[pi];
            auto& m = m_[pi];
            auto& v = v_[pi];
            for (size_t i = 0; i < p->size(); ++i) {
                float g = p->grad[i];
                m[i] = cfg_.beta1 * m[i] + (1.0f - cfg_.beta1) * g;
                v[i] = cfg_.beta2 * v[i] + (1.0f - cfg_.beta2) * g * g;
                float m_hat = m[i] / (1.0f - beta1t);
                float v_hat = v[i] / (1.0f - beta2t);
                p->data[i] -= cfg_.lr * m_hat / (std::sqrt(v_hat) + cfg_.eps);
            }
        }
    } else {
        for (auto* p : params_) {
            for (size_t i = 0; i < p->size(); ++i) {
                p->data[i] -= cfg_.lr * p->grad[i];
            }
        }
    }
}

void Optimizer::set_state(const std::vector<std::vector<float>>& m,
                          const std::vector<std::vector<float>>& v,
                          size_t t) {
    if (!cfg_.use_adam) return;
    if (m.size() == params_.size() && v.size() == params_.size()) {
        m_ = m;
        v_ = v;
        t_ = t;
    }
}
