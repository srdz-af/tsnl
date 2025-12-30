#pragma once

#include "csr.hpp"
#include "rng.hpp"
#include "sampler.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

struct LayerSamples {
    std::vector<uint32_t> offsets;  // len = targets+1
    std::vector<uint32_t> neighbors;
    std::vector<uint16_t> rels;
};

struct BatchSubgraph {
    std::vector<std::vector<uint32_t>> nodes_per_layer; // 0 is farthest
    std::vector<LayerSamples> samples;                  // size = L
};

BatchSubgraph build_subgraph(const CsrGraph& g, const std::vector<uint32_t>& seeds,
                             const std::vector<size_t>& fanouts, XorShift128Plus& rng);
