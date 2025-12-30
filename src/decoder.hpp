#pragma once

#include "encoder.hpp"
#include "optim.hpp"

#include <unordered_map>
#include <vector>

class Decoder {
public:
    Decoder(size_t num_relations, size_t dim, Parameter* shared_rel_emb, XorShift128Plus& rng);

    float distmult_loss(const std::vector<uint32_t>& heads,
                        const std::vector<uint32_t>& rels,
                        const std::vector<uint32_t>& tails,
                        const std::vector<uint32_t>& neg_tails,
                        size_t neg_per_pos,
                        const std::unordered_map<uint32_t, size_t>& index_map,
                        const std::vector<float>& embeddings,
                        std::vector<float>& grad_out);

    float relation_loss(const std::vector<uint32_t>& heads,
                        const std::vector<uint32_t>& tails,
                        const std::vector<uint32_t>& rels,
                        const std::unordered_map<uint32_t, size_t>& index_map,
                        const std::vector<float>& embeddings,
                        std::vector<float>& grad_out,
                        float weight = 1.0f);

    std::vector<Parameter*> parameters();
    std::vector<const Parameter*> parameters_const() const;
    const Parameter& rel_cls_w() const { return rel_cls_w_; }
    const Parameter& rel_cls_b() const { return rel_cls_b_; }

private:
    size_t num_rel_;
    size_t dim_;
    Parameter* rel_emb_; // shared with encoder
    Parameter rel_cls_w_; // (num_rel_+1) x 4*dim_
    Parameter rel_cls_b_; // (num_rel_+1)
};
