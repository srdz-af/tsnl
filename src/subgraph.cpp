#include "subgraph.hpp"

#include <unordered_set>

static std::vector<uint32_t> dedup_nodes(const std::vector<uint32_t>& nodes) {
    std::unordered_set<uint32_t> set;
    set.reserve(nodes.size() * 2 + 1);
    for (uint32_t v : nodes) set.insert(v);
    std::vector<uint32_t> out;
    out.reserve(set.size());
    for (uint32_t v : set) out.push_back(v);
    return out;
}

BatchSubgraph build_subgraph(const CsrGraph& g, const std::vector<uint32_t>& seeds,
                             const std::vector<size_t>& fanouts, XorShift128Plus& rng) {
    BatchSubgraph sg;
    const size_t L = fanouts.size();
    sg.nodes_per_layer.resize(L + 1);
    sg.samples.resize(L);
    sg.nodes_per_layer[L] = dedup_nodes(seeds);

    for (int l = static_cast<int>(L) - 1; l >= 0; --l) {
        const auto& targets = sg.nodes_per_layer[l + 1];
        LayerSamples ls;
        ls.offsets.resize(targets.size() + 1);
        ls.offsets[0] = 0;
        for (size_t i = 0; i < targets.size(); ++i) {
            uint32_t v = targets[i];
            sample_neighbors(g, v, fanouts[l], ls.neighbors, ls.rels, rng);
            ls.offsets[i + 1] = static_cast<uint32_t>(ls.neighbors.size());
        }
        sg.samples[l] = std::move(ls);

        std::vector<uint32_t> prev_nodes;
        prev_nodes.reserve(sg.nodes_per_layer[l + 1].size() + sg.samples[l].neighbors.size());
        for (uint32_t v : sg.nodes_per_layer[l + 1]) prev_nodes.push_back(v);
        for (uint32_t n : sg.samples[l].neighbors) prev_nodes.push_back(n);
        sg.nodes_per_layer[l] = dedup_nodes(prev_nodes);
    }

    return sg;
}
