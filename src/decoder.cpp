#include "decoder.hpp"

#include <algorithm>
#include <cmath>

static void init_param(Parameter& p, size_t n, XorShift128Plus& rng, float scale) {
    p = Parameter(n);
    for (size_t i = 0; i < n; ++i) {
        float r = static_cast<float>(rng.uniform() - 0.5f);
        p.data[i] = r * scale;
    }
}

Decoder::Decoder(size_t num_relations, size_t dim, Parameter* shared_rel_emb, XorShift128Plus& rng)
    : num_rel_(num_relations), dim_(dim), rel_emb_(shared_rel_emb) {
    init_param(rel_cls_w_, (num_rel_ + 1) * 4 * dim_, rng, 0.1f / std::sqrt(static_cast<float>(dim_)));
    init_param(rel_cls_b_, num_rel_ + 1, rng, 0.01f);
}

float Decoder::distmult_loss(const std::vector<uint32_t>& heads,
                             const std::vector<uint32_t>& rels,
                             const std::vector<uint32_t>& tails,
                             const std::vector<uint32_t>& neg_tails,
                             size_t neg_per_pos,
                             const std::unordered_map<uint32_t, size_t>& index_map,
                             const std::vector<float>& embeddings,
                             std::vector<float>& grad_out) {
    float loss = 0.0f;
    size_t batch = heads.size();
    for (size_t i = 0; i < batch; ++i) {
        auto hit = index_map.find(heads[i]);
        auto tit = index_map.find(tails[i]);
        if (hit == index_map.end() || tit == index_map.end()) continue;
        size_t h_idx = hit->second;
        size_t t_idx = tit->second;
        const float* h = &embeddings[h_idx * dim_];
        const float* t = &embeddings[t_idx * dim_];
        const float* rvec = &rel_emb_->data[static_cast<size_t>(rels[i]) * dim_];
        float score = 0.0f;
        for (size_t d = 0; d < dim_; ++d) score += h[d] * rvec[d] * t[d];
        float pos_grad = -1.0f / (1.0f + std::exp(score));
        loss += std::log1pf(std::exp(-score));
        for (size_t d = 0; d < dim_; ++d) {
            float common = rvec[d] * t[d];
            grad_out[h_idx * dim_ + d] += pos_grad * common;
            grad_out[t_idx * dim_ + d] += pos_grad * rvec[d] * h[d];
            rel_emb_->grad[static_cast<size_t>(rels[i]) * dim_ + d] += pos_grad * h[d] * t[d];
        }

        const uint32_t* negs = &neg_tails[i * neg_per_pos];
        for (size_t k = 0; k < neg_per_pos; ++k) {
            auto nit = index_map.find(negs[k]);
            if (nit == index_map.end()) continue;
            size_t n_idx = nit->second;
            const float* nvec = &embeddings[n_idx * dim_];
            float sneg = 0.0f;
            for (size_t d = 0; d < dim_; ++d) sneg += h[d] * rvec[d] * nvec[d];
            float neg_grad = 1.0f / (1.0f + std::exp(-sneg));
            loss += std::log1pf(std::exp(sneg));
            for (size_t d = 0; d < dim_; ++d) {
                float common = rvec[d] * nvec[d];
                grad_out[h_idx * dim_ + d] += neg_grad * common;
                grad_out[n_idx * dim_ + d] += neg_grad * rvec[d] * h[d];
                rel_emb_->grad[static_cast<size_t>(rels[i]) * dim_ + d] += neg_grad * h[d] * nvec[d];
            }
        }
    }
    if (batch > 0) loss /= static_cast<float>(batch);
    return loss;
}

float Decoder::relation_loss(const std::vector<uint32_t>& heads,
                             const std::vector<uint32_t>& tails,
                             const std::vector<uint32_t>& rels,
                             const std::unordered_map<uint32_t, size_t>& index_map,
                             const std::vector<float>& embeddings,
                             std::vector<float>& grad_out,
                             float weight) {
    float loss = 0.0f;
    size_t batch = heads.size();
    const size_t phi_dim = 4 * dim_;
    std::vector<float> phi(phi_dim, 0.0f);
    std::vector<float> grad_phi(phi_dim, 0.0f);
    std::vector<float> logits(num_rel_ + 1, 0.0f);

    for (size_t i = 0; i < batch; ++i) {
        auto hit = index_map.find(heads[i]);
        auto tit = index_map.find(tails[i]);
        if (hit == index_map.end() || tit == index_map.end()) continue;
        size_t h_idx = hit->second;
        size_t t_idx = tit->second;
        const float* h = &embeddings[h_idx * dim_];
        const float* t = &embeddings[t_idx * dim_];

        for (size_t d = 0; d < dim_; ++d) {
            phi[d] = h[d];
            phi[dim_ + d] = t[d];
            phi[2 * dim_ + d] = h[d] * t[d];
            phi[3 * dim_ + d] = std::abs(h[d] - t[d]);
        }
        // logits
        float max_logit = -1e9f;
        for (size_t r = 1; r <= num_rel_; ++r) {
            float s = rel_cls_b_.data[r];
            const float* w = &rel_cls_w_.data[r * phi_dim];
            for (size_t k = 0; k < phi_dim; ++k) s += w[k] * phi[k];
            logits[r] = s;
            if (s > max_logit) max_logit = s;
        }
        float denom = 0.0f;
        for (size_t r = 1; r <= num_rel_; ++r) denom += std::exp(logits[r] - max_logit);
        float log_denom = std::log(denom) + max_logit;
        uint32_t gold = rels[i];
        loss += (-logits[gold] + log_denom) * weight;

        // gradients
        for (size_t k = 0; k < phi_dim; ++k) grad_phi[k] = 0.0f;
        for (size_t r = 1; r <= num_rel_; ++r) {
            float prob = std::exp(logits[r] - log_denom);
            float g = (prob - (r == gold ? 1.0f : 0.0f)) * weight;
            rel_cls_b_.grad[r] += g;
            float* gw = &rel_cls_w_.grad[r * phi_dim];
            const float* w = &rel_cls_w_.data[r * phi_dim];
            for (size_t k = 0; k < phi_dim; ++k) {
                gw[k] += g * phi[k];
                grad_phi[k] += g * w[k];
            }
        }

        float* gh = &grad_out[h_idx * dim_];
        float* gt = &grad_out[t_idx * dim_];
        for (size_t d = 0; d < dim_; ++d) {
            gh[d] += grad_phi[d];
            gt[d] += grad_phi[dim_ + d];
            gh[d] += grad_phi[2 * dim_ + d] * t[d];
            gt[d] += grad_phi[2 * dim_ + d] * h[d];
            float sign = (h[d] >= t[d]) ? 1.0f : -1.0f;
            gh[d] += grad_phi[3 * dim_ + d] * sign;
            gt[d] -= grad_phi[3 * dim_ + d] * sign;
        }
    }
    if (batch > 0) loss /= static_cast<float>(batch);
    return loss;
}

std::vector<Parameter*> Decoder::parameters() {
    return {&rel_cls_w_, &rel_cls_b_};
}

std::vector<const Parameter*> Decoder::parameters_const() const {
    return {&rel_cls_w_, &rel_cls_b_};
}
