#include "features.hpp"

#include <cmath>

size_t feature_dim(const FeatureConfig& cfg, bool has_reverse) {
    (void)has_reverse;
    size_t dim = 1;
    if (cfg.use_in_degree) dim += 1;
    return dim;
}

void compute_base_features(const CsrGraph& g, const CsrGraph* rev,
                           const std::vector<uint32_t>& nodes,
                           const FeatureConfig& cfg,
                           std::vector<float>& out) {
    const bool has_rev = (rev != nullptr);
    size_t dim = feature_dim(cfg, has_rev);
    out.assign(nodes.size() * dim, 0.0f);
    for (size_t i = 0; i < nodes.size(); ++i) {
        uint32_t v = nodes[i];
        size_t idx = i * dim;
        float deg_out = static_cast<float>(g.out_degree(v));
        out[idx] = std::log1pf(deg_out);
        if (cfg.use_in_degree) {
            float deg_in = has_rev ? static_cast<float>(rev->out_degree(v)) : 0.0f;
            out[idx + 1] = std::log1pf(deg_in);
        }
        if (cfg.add_noise) {
            uint64_t noise_seed = mix_seed(static_cast<uint64_t>(v));
            float noise = static_cast<float>((noise_seed & 0xFFFF) / 65535.0 - 0.5) * 0.02f;
            for (size_t d = 0; d < dim; ++d) out[idx + d] += noise;
        }
    }
}
