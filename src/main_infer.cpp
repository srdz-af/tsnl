#include "checkpoint.hpp"
#include "csr.hpp"
#include "decoder.hpp"
#include "encoder.hpp"
#include "features.hpp"
#include "io.hpp"
#include "rng.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

struct InferOptions {
    std::string data_dir = "data";
    std::string reverse_dir;
    std::string checkpoint;
    std::string relation_queries;
    std::string tail_queries;
    size_t topk = 5;
    size_t batch_nodes = 1024;
    uint64_t seed = 123;
};

static bool parse_args(int argc, char** argv, InferOptions& opt) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](int more) { return i + more < argc; };
        if ((a == "--data" || a == "-d") && need(1)) {
            opt.data_dir = argv[++i];
        } else if (a == "--reverse" && need(1)) {
            opt.reverse_dir = argv[++i];
        } else if ((a == "--checkpoint" || a == "-c") && need(1)) {
            opt.checkpoint = argv[++i];
        } else if (a == "--relation_queries" && need(1)) {
            opt.relation_queries = argv[++i];
        } else if (a == "--tail_queries" && need(1)) {
            opt.tail_queries = argv[++i];
        } else if (a == "--topk" && need(1)) {
            opt.topk = std::stoul(argv[++i]);
        } else if (a == "--batch_nodes" && need(1)) {
            opt.batch_nodes = std::stoul(argv[++i]);
        } else if (a == "--seed" && need(1)) {
            opt.seed = std::stoull(argv[++i]);
        } else {
            return false;
        }
    }
    return !opt.checkpoint.empty();
}

static std::vector<float> build_cache(Encoder& enc, const CsrGraph& g, const CsrGraph* rev,
                                      size_t dim, size_t batch_nodes, XorShift128Plus& rng) {
    std::vector<float> cache(static_cast<size_t>(g.num_nodes()) * dim, 0.0f);
    for (uint32_t start = 1; start <= g.num_nodes(); start += batch_nodes) {
        uint32_t end = std::min<uint32_t>(g.num_nodes(), start + batch_nodes - 1);
        std::vector<uint32_t> seeds;
        seeds.reserve(end - start + 1);
        for (uint32_t v = start; v <= end; ++v) seeds.push_back(v);
        EncoderState st = enc.forward(g, rev, seeds, rng);
        const auto& map = st.index_per_layer.back();
        const auto& emb = st.h_layers.back();
        for (uint32_t v : seeds) {
            auto it = map.find(v);
            if (it == map.end()) continue;
            size_t idx = it->second;
            std::memcpy(&cache[(static_cast<size_t>(v - 1)) * dim],
                        &emb[idx * dim],
                        sizeof(float) * dim);
        }
    }
    return cache;
}

static void run_relation_queries(const InferOptions& opt, Decoder& dec, size_t dim,
                                 const std::vector<float>& cache, const MMapArray<Triple>& queries) {
    const size_t phi_dim = 4 * dim;
    std::vector<float> phi(phi_dim, 0.0f);
    for (size_t i = 0; i < queries.size; ++i) {
        const Triple& q = queries[i];
        if (q.h == 0 || q.t == 0) continue;
        const float* uh = &cache[(static_cast<size_t>(q.h - 1)) * dim];
        const float* tv = &cache[(static_cast<size_t>(q.t - 1)) * dim];
        for (size_t d = 0; d < dim; ++d) {
            phi[d] = uh[d];
            phi[dim + d] = tv[d];
            phi[2 * dim + d] = uh[d] * tv[d];
            phi[3 * dim + d] = std::abs(uh[d] - tv[d]);
        }
        const auto& W = dec.rel_cls_w().data;
        const auto& B = dec.rel_cls_b().data;
        std::vector<float> scores(dec.rel_cls_b().size(), 0.0f);
        float maxv = -1e9f;
        for (size_t r = 1; r < scores.size(); ++r) {
            float s = B[r];
            const float* w = &W[r * phi_dim];
            for (size_t k = 0; k < phi_dim; ++k) s += w[k] * phi[k];
            scores[r] = s;
            if (s > maxv) maxv = s;
        }
        std::vector<std::pair<float, size_t>> top;
        top.reserve(scores.size() - 1);
        for (size_t r = 1; r < scores.size(); ++r) top.emplace_back(scores[r], r);
        std::partial_sort(top.begin(), top.begin() + std::min(opt.topk, top.size()), top.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        std::cout << "Query (" << q.h << "," << q.t << ") top relations: ";
        size_t limit = std::min(opt.topk, top.size());
        for (size_t k = 0; k < limit; ++k) {
            std::cout << top[k].second << ":" << top[k].first << " ";
        }
        if (q.r != 0) {
            float true_score = scores[q.r];
            size_t rank = 1;
            for (size_t r = 1; r < scores.size(); ++r) {
                if (scores[r] > true_score) ++rank;
            }
            std::cout << " true_r=" << q.r << " rank=" << rank;
        }
        std::cout << "\n";
    }
}

static void run_tail_queries(const InferOptions& opt, Encoder& enc, size_t dim,
                             const std::vector<float>& cache, const MMapArray<Triple>& queries) {
    const float* rel_emb = enc.relation_embeddings()->data.data();
    for (size_t i = 0; i < queries.size; ++i) {
        const Triple& q = queries[i];
        if (q.h == 0 || q.r == 0) continue;
        const float* h = &cache[(static_cast<size_t>(q.h - 1)) * dim];
        const float* rvec = &rel_emb[static_cast<size_t>(q.r) * dim];
        std::vector<float> scores(cache.size() / dim, 0.0f);
        float true_score = 0.0f;
        for (size_t v = 1; v <= scores.size(); ++v) {
            const float* t = &cache[(v - 1) * dim];
            float s = 0.0f;
            for (size_t d = 0; d < dim; ++d) s += h[d] * rvec[d] * t[d];
            scores[v - 1] = s;
            if (v == q.t) true_score = s;
        }
        size_t rank = 1;
        for (float s : scores) if (s > true_score) ++rank;
        std::vector<size_t> idx(scores.size());
        for (size_t v = 0; v < scores.size(); ++v) idx[v] = v + 1;
        std::partial_sort(idx.begin(), idx.begin() + std::min(opt.topk, idx.size()), idx.end(),
                          [&](size_t a, size_t b) { return scores[a - 1] > scores[b - 1]; });
        std::cout << "Query (" << q.h << "," << q.r << ",?) top tails: ";
        size_t limit = std::min(opt.topk, idx.size());
        for (size_t k = 0; k < limit; ++k) {
            std::cout << idx[k] << ":" << scores[idx[k] - 1] << " ";
        }
        std::cout << " true_t=" << q.t << " rank=" << rank << "\n";
    }
}

int main(int argc, char** argv) {
    InferOptions opt;
    if (!parse_args(argc, argv, opt)) {
        std::cerr << "Invalid arguments\n";
        return 1;
    }

    CsrGraph g(opt.data_dir);
    if (!g.valid()) {
        std::cerr << "Failed to load CSR graph\n";
        return 1;
    }
    CsrGraph* rev_ptr = nullptr;
    CsrGraph rev;
    if (!opt.reverse_dir.empty()) {
        if (rev.load_custom(opt.reverse_dir, "offsets_rev.bin", "csr_rev.bin", "rels_rev.bin")) {
            rev_ptr = &rev;
        }
    }

    CheckpointMeta meta;
    std::vector<std::vector<float>> params, m, v;
    if (!load_checkpoint(opt.checkpoint, meta, params, m, v)) {
        std::cerr << "Failed to load checkpoint\n";
        return 1;
    }

    FeatureConfig fcfg = meta.feat_cfg;
    if (meta.enc_cfg.fanouts.empty()) {
        meta.enc_cfg.fanouts.resize(meta.enc_cfg.layers, 10);
    }

    XorShift128Plus rng(opt.seed);
    Encoder encoder(meta.feature_dim ? meta.feature_dim : feature_dim(fcfg, rev_ptr != nullptr),
                    g.num_relations(), meta.enc_cfg, fcfg, rng);
    Decoder decoder(g.num_relations(), meta.enc_cfg.hidden_dim, encoder.relation_embeddings(), rng);
    auto enc_params = encoder.parameters();
    auto dec_params = decoder.parameters();
    std::vector<Parameter*> all_params = enc_params;
    all_params.insert(all_params.end(), dec_params.begin(), dec_params.end());
    assign_parameters(params, all_params);
    if (!m.empty() && !v.empty()) {
        OptimConfig oc;
        oc.use_adam = meta.use_adam;
        Optimizer opttmp(oc, all_params);
        opttmp.set_state(m, v, meta.step);
    }

    std::vector<float> cache = build_cache(encoder, g, rev_ptr, encoder.output_dim(), opt.batch_nodes, rng);

    if (!opt.relation_queries.empty()) {
        MMapArray<Triple> q;
        if (map_triples(opt.relation_queries, q)) {
            run_relation_queries(opt, decoder, encoder.output_dim(), cache, q);
        }
    }
    if (!opt.tail_queries.empty()) {
        MMapArray<Triple> q;
        if (map_triples(opt.tail_queries, q)) {
            run_tail_queries(opt, encoder, encoder.output_dim(), cache, q);
        }
    }

    return 0;
}
