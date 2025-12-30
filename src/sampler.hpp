#pragma once

#include "csr.hpp"
#include "rng.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

void sample_neighbors(const CsrGraph& g, uint32_t node, size_t fanout,
                      std::vector<uint32_t>& out_nodes, std::vector<uint16_t>& out_rels,
                      XorShift128Plus& rng);

uint32_t sample_negative(uint32_t num_nodes, XorShift128Plus& rng);
