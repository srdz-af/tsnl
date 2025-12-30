#include "encoder.hpp"

#include "features.hpp"
#include "sampler.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

static void init_param(Parameter& p, size_t n, XorShift128Plus& rng, float scale) {
    p = Parameter(n);
    for (size_t i = 0; i < n; ++i) {
        float r = static_cast<float>(rng.uniform() - 0.5f);
        p.data[i] = r * scale;
    }
}

Encoder::Encoder(size_t feature_dim, size_t num_relations, const EncoderConfig& cfg,
                 const FeatureConfig& feat_cfg, XorShift128Plus& rng)
    : cfg_(cfg), feature_dim_(feature_dim), num_rel_(num_relations), feat_cfg_(feat_cfg) {
    if (feature_dim_ <= 1) feat_cfg_.use_in_degree = false;
    init_param(input_w_, feature_dim_ * cfg_.hidden_dim, rng, 0.1f);
    init_param(input_b_, cfg_.hidden_dim, rng, 0.01f);
    layer_w_.resize(cfg_.layers);
    layer_b_.resize(cfg_.layers);
    for (int l = 0; l < cfg_.layers; ++l) {
        init_param(layer_w_[l], 2 * cfg_.hidden_dim * cfg_.hidden_dim, rng, 0.1f / std::sqrt(static_cast<float>(cfg_.hidden_dim)));
        init_param(layer_b_[l], cfg_.hidden_dim, rng, 0.01f);
    }
    init_param(rel_emb_, (num_rel_ + 1) * cfg_.hidden_dim, rng, 0.1f);
}

EncoderState Encoder::forward(const CsrGraph& g, const CsrGraph* rev,
                              const std::vector<uint32_t>& batch_nodes, XorShift128Plus& rng) {
    EncoderState st;
    st.sg = build_subgraph(g, batch_nodes, cfg_.fanouts, rng);
    size_t L = cfg_.fanouts.size();

    st.index_per_layer.resize(L + 1);
    for (size_t l = 0; l <= L; ++l) {
        auto& map = st.index_per_layer[l];
        const auto& nodes = st.sg.nodes_per_layer[l];
        map.reserve(nodes.size() * 2 + 1);
        for (size_t i = 0; i < nodes.size(); ++i) map[nodes[i]] = i;
    }

    st.h_layers.resize(L + 1);
    st.pre_layers.resize(L + 1);
    st.agg_layers.resize(L);

    // Base features
    st.base_features.clear();
    compute_base_features(g, rev, st.sg.nodes_per_layer[0], feat_cfg_, st.base_features);

    // Input projection
    const size_t hidden = cfg_.hidden_dim;
    const size_t n0 = st.sg.nodes_per_layer[0].size();
    st.h_layers[0].resize(n0 * hidden);
    st.pre_layers[0].resize(n0 * hidden);
    for (size_t i = 0; i < n0; ++i) {
        const float* feat = &st.base_features[i * feature_dim_];
        for (size_t d = 0; d < hidden; ++d) {
            float sum = input_b_.data[d];
            for (size_t f = 0; f < feature_dim_; ++f) {
                sum += feat[f] * input_w_.data[f * hidden + d];
            }
            st.pre_layers[0][i * hidden + d] = sum;
            st.h_layers[0][i * hidden + d] = cfg_.use_relu ? std::max(0.0f, sum) : sum;
        }
    }

    // Aggregation layers
    for (size_t l = 0; l < L; ++l) {
        const auto& targets = st.sg.nodes_per_layer[l + 1];
        const auto& map_l = st.index_per_layer[l];
        const LayerSamples& ls = st.sg.samples[l];
        st.agg_layers[l].assign(targets.size() * hidden, 0.0f);
        st.pre_layers[l + 1].resize(targets.size() * hidden);
        st.h_layers[l + 1].resize(targets.size() * hidden);

        for (size_t ti = 0; ti < targets.size(); ++ti) {
            uint32_t v = targets[ti];
            auto it = map_l.find(v);
            size_t self_idx = (it == map_l.end()) ? static_cast<size_t>(-1) : it->second;
            const float* self = self_idx == static_cast<size_t>(-1) ? nullptr : &st.h_layers[l][self_idx * hidden];

            uint32_t start = ls.offsets[ti];
            uint32_t end = ls.offsets[ti + 1];
            size_t deg = (end > start) ? (end - start) : 0;
            float* agg = &st.agg_layers[l][ti * hidden];
            if (deg > 0) {
                for (uint32_t e = start; e < end; ++e) {
                    uint32_t nb = ls.neighbors[e];
                    uint16_t rel = ls.rels[e];
                    auto nb_it = map_l.find(nb);
                    if (nb_it == map_l.end()) continue;
                    const float* nb_vec = &st.h_layers[l][nb_it->second * hidden];
                    const float* rel_vec = &rel_emb_.data[static_cast<size_t>(rel) * hidden];
                    for (size_t d = 0; d < hidden; ++d) {
                        agg[d] += nb_vec[d] + rel_vec[d];
                    }
                }
                float inv = 1.0f / static_cast<float>(deg);
                for (size_t d = 0; d < hidden; ++d) agg[d] *= inv;
            } else {
                std::fill(agg, agg + hidden, 0.0f);
            }

            float* pre = &st.pre_layers[l + 1][ti * hidden];
            float* out = &st.h_layers[l + 1][ti * hidden];
            for (size_t d = 0; d < hidden; ++d) {
                float sum = layer_b_[l].data[d];
                for (size_t k = 0; k < hidden; ++k) {
                    float self_val = self ? self[k] : 0.0f;
                    sum += self_val * layer_w_[l].data[k * hidden + d]; // self part
                    sum += agg[k] * layer_w_[l].data[(hidden + k) * hidden + d]; // agg part
                }
                pre[d] = sum;
                out[d] = cfg_.use_relu ? std::max(0.0f, sum) : sum;
            }
        }
    }

    return st;
}

void Encoder::backward(EncoderState& st, std::vector<std::vector<float>>& grad_layers) {
    size_t L = cfg_.fanouts.size();
    const size_t hidden = cfg_.hidden_dim;
    if (grad_layers.size() != L + 1) grad_layers.resize(L + 1);
    // Allocate grad for lower layers
    for (size_t l = 0; l < L; ++l) {
        grad_layers[l].assign(st.sg.nodes_per_layer[l].size() * hidden, 0.0f);
    }

    // Backprop through aggregation layers
    for (int l = static_cast<int>(L) - 1; l >= 0; --l) {
        const auto& targets = st.sg.nodes_per_layer[l + 1];
        const auto& map_l = st.index_per_layer[l];
        const LayerSamples& ls = st.sg.samples[l];
        auto& grad_out = grad_layers[l + 1];
        for (size_t ti = 0; ti < targets.size(); ++ti) {
            float* grad_pre = &grad_out[ti * hidden];
            // ReLU backprop
            if (cfg_.use_relu) {
                const float* pre = &st.pre_layers[l + 1][ti * hidden];
                for (size_t d = 0; d < hidden; ++d) {
                    if (pre[d] <= 0.0f) grad_pre[d] = 0.0f;
                }
            }
            uint32_t v = targets[ti];
            auto it = map_l.find(v);
            if (it == map_l.end()) continue;
            size_t self_idx = it->second;
            const float* self = &st.h_layers[l][self_idx * hidden];
            const float* agg = &st.agg_layers[l][ti * hidden];

            // Gradient w.r.t weights and bias
            for (size_t d_out = 0; d_out < hidden; ++d_out) {
                float g = grad_pre[d_out];
                layer_b_[l].grad[d_out] += g;
                for (size_t k = 0; k < hidden; ++k) {
                    float self_val = self ? self[k] : 0.0f;
                    layer_w_[l].grad[k * hidden + d_out] += g * self_val;
                    layer_w_[l].grad[(hidden + k) * hidden + d_out] += g * agg[k];
                }
            }

            // Gradients propagated to concat inputs
            std::vector<float> grad_concat(2 * hidden, 0.0f);
            for (size_t k = 0; k < 2 * hidden; ++k) {
                float sum = 0.0f;
                for (size_t d_out = 0; d_out < hidden; ++d_out) {
                    sum += grad_pre[d_out] * layer_w_[l].data[k * hidden + d_out];
                }
                grad_concat[k] = sum;
            }
            float* grad_self = &grad_layers[l][self_idx * hidden];
            for (size_t k = 0; k < hidden; ++k) grad_self[k] += grad_concat[k];

            uint32_t start = ls.offsets[ti];
            uint32_t end = ls.offsets[ti + 1];
            size_t deg = (end > start) ? (end - start) : 0;
            if (deg == 0) continue;
            float inv = 1.0f / static_cast<float>(deg);
            for (uint32_t e = start; e < end; ++e) {
                uint32_t nb = ls.neighbors[e];
                uint16_t rel = ls.rels[e];
                auto nb_it = map_l.find(nb);
                if (nb_it == map_l.end()) continue;
                size_t nb_idx = nb_it->second;
                float* grad_nb = &grad_layers[l][nb_idx * hidden];
                for (size_t d = 0; d < hidden; ++d) {
                    float gshare = grad_concat[hidden + d] * inv;
                    grad_nb[d] += gshare;
                    rel_emb_.grad[static_cast<size_t>(rel) * hidden + d] += gshare;
                }
            }
        }
    }

    // Backprop input projection
    {
        const size_t n0 = st.sg.nodes_per_layer[0].size();
        auto& grad0 = grad_layers[0];
        for (size_t i = 0; i < n0; ++i) {
            const float* pre = &st.pre_layers[0][i * hidden];
            for (size_t d = 0; d < hidden; ++d) {
                if (cfg_.use_relu && pre[d] <= 0.0f) grad0[i * hidden + d] = 0.0f;
            }
        }
        for (size_t i = 0; i < n0; ++i) {
            const float* feat = &st.base_features[i * feature_dim_];
            float* gvec = &grad0[i * hidden];
            for (size_t d = 0; d < hidden; ++d) {
                float g = gvec[d];
                input_b_.grad[d] += g;
                for (size_t f = 0; f < feature_dim_; ++f) {
                    input_w_.grad[f * hidden + d] += g * feat[f];
                }
            }
        }
    }
}

std::vector<Parameter*> Encoder::parameters() {
    std::vector<Parameter*> ps;
    ps.push_back(&input_w_);
    ps.push_back(&input_b_);
    for (auto& w : layer_w_) ps.push_back(&w);
    for (auto& b : layer_b_) ps.push_back(&b);
    ps.push_back(&rel_emb_);
    return ps;
}

std::vector<const Parameter*> Encoder::parameters_const() const {
    std::vector<const Parameter*> ps;
    ps.push_back(&input_w_);
    ps.push_back(&input_b_);
    for (const auto& w : layer_w_) ps.push_back(&w);
    for (const auto& b : layer_b_) ps.push_back(&b);
    ps.push_back(&rel_emb_);
    return ps;
}
