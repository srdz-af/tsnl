#pragma once

#include <cstddef>
#include <vector>

struct Parameter {
    std::vector<float> data;
    std::vector<float> grad;

    Parameter() = default;
    explicit Parameter(size_t n, float init = 0.0f);
    void zero_grad();
    size_t size() const { return data.size(); }
};

struct OptimConfig {
    float lr = 0.001f;
    bool use_adam = true;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
};

class Optimizer {
public:
    Optimizer(const OptimConfig& cfg, const std::vector<Parameter*>& params);
    void step();
    void zero_grad();
    const std::vector<std::vector<float>>& m() const { return m_; }
    const std::vector<std::vector<float>>& v() const { return v_; }
    size_t step_count() const { return t_; }
    void set_state(const std::vector<std::vector<float>>& m,
                   const std::vector<std::vector<float>>& v,
                   size_t t);

private:
    OptimConfig cfg_;
    std::vector<Parameter*> params_;
    std::vector<std::vector<float>> m_;
    std::vector<std::vector<float>> v_;
    size_t t_ = 0;
};
