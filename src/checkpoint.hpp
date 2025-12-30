#pragma once

#include "decoder.hpp"
#include "encoder.hpp"
#include "optim.hpp"

#include <string>
#include <vector>

struct CheckpointMeta {
    EncoderConfig enc_cfg;
    FeatureConfig feat_cfg;
    size_t num_rel = 0;
    size_t feature_dim = 0;
    bool use_adam = true;
    size_t step = 0;
};

bool save_checkpoint(const std::string& path, const Encoder& enc, const Decoder& dec,
                     const FeatureConfig& feat_cfg, const Optimizer* opt);

bool load_checkpoint(const std::string& path, CheckpointMeta& meta,
                     std::vector<std::vector<float>>& params_out,
                     std::vector<std::vector<float>>& m_out,
                     std::vector<std::vector<float>>& v_out);

void assign_parameters(const std::vector<std::vector<float>>& data,
                       const std::vector<Parameter*>& params);
