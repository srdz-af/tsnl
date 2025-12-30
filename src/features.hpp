#pragma once

#include "csr.hpp"
#include "rng.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

struct FeatureConfig {
    bool use_in_degree = true;
    bool add_noise = false;
};

size_t feature_dim(const FeatureConfig& cfg, bool has_reverse);

void compute_base_features(const CsrGraph& g, const CsrGraph* rev,
                           const std::vector<uint32_t>& nodes,
                           const FeatureConfig& cfg,
                           std::vector<float>& out);
