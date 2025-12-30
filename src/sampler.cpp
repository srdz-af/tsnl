#include "sampler.hpp"

void sample_neighbors(const CsrGraph& g, uint32_t node, size_t fanout,
                      std::vector<uint32_t>& out_nodes, std::vector<uint16_t>& out_rels,
                      XorShift128Plus& rng) {
    AdjView adj = g.neighbors(node);
    if (adj.size == 0 || fanout == 0) return;
    out_nodes.reserve(out_nodes.size() + fanout);
    out_rels.reserve(out_rels.size() + fanout);
    for (size_t i = 0; i < fanout; ++i) {
        uint32_t idx = rng.next_u32(adj.size);
        out_nodes.push_back(adj.dst[idx]);
        out_rels.push_back(adj.rel[idx]);
    }
}

uint32_t sample_negative(uint32_t num_nodes, XorShift128Plus& rng) {
    uint32_t v = rng.next_u32(num_nodes) + 1;
    return v;
}
