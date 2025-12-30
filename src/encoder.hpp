#pragma once

#include "csr.hpp"
#include "features.hpp"
#include "optim.hpp"
#include "subgraph.hpp"

#include <unordered_map>
#include <vector>

struct EncoderConfig {
    size_t hidden_dim = 64;
    int layers = 2;
    std::vector<size_t> fanouts{20, 10};
    bool use_relu = true;
};

struct EncoderState {
    BatchSubgraph sg;
    std::vector<std::vector<float>> h_layers;   // length = L+1
    std::vector<std::vector<float>> pre_layers; // pre-activation, length = L+1
    std::vector<std::vector<float>> agg_layers; // length = L
    std::vector<std::unordered_map<uint32_t, size_t>> index_per_layer;
    std::vector<float> base_features;
};

class Encoder {
public:
    Encoder(size_t feature_dim, size_t num_relations, const EncoderConfig& cfg,
            const FeatureConfig& feat_cfg, XorShift128Plus& rng);

    EncoderState forward(const CsrGraph& g, const CsrGraph* rev,
                         const std::vector<uint32_t>& batch_nodes, XorShift128Plus& rng);

    void backward(EncoderState& state, std::vector<std::vector<float>>& grad_layers);

    std::vector<Parameter*> parameters();
    std::vector<const Parameter*> parameters_const() const;
    size_t output_dim() const { return cfg_.hidden_dim; }
    size_t num_relations() const { return num_rel_; }
    const EncoderConfig& config() const { return cfg_; }
    size_t feature_dim_raw() const { return feature_dim_; }
    Parameter* relation_embeddings() { return &rel_emb_; }

private:
    EncoderConfig cfg_;
    size_t feature_dim_;
    size_t num_rel_;
    FeatureConfig feat_cfg_;
    Parameter input_w_;
    Parameter input_b_;
    std::vector<Parameter> layer_w_;
    std::vector<Parameter> layer_b_;
    Parameter rel_emb_; // (num_rel_+1) x hidden_dim
};
