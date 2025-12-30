#include "checkpoint.hpp"
#include "csr.hpp"
#include "decoder.hpp"
#include "encoder.hpp"
#include "io.hpp"
#include "metrics.hpp"
#include "features.hpp"
#include "rng.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct EvalOptions {
    std::string data_dir = "data";
    std::string reverse_dir;
    std::string checkpoint;
    std::string eval_file;
    std::string train_file;
    size_t batch_nodes = 1024;
    uint64_t seed = 99;
};

static bool parse_args(int argc, char** argv, EvalOptions& opt) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](int more) { return i + more < argc; };
        if ((a == "--data" || a == "-d") && need(1)) {
            opt.data_dir = argv[++i];
        } else if (a == "--reverse" && need(1)) {
            opt.reverse_dir = argv[++i];
        } else if ((a == "--checkpoint" || a == "-c") && need(1)) {
            opt.checkpoint = argv[++i];
        } else if ((a == "--eval" || a == "-e") && need(1)) {
            opt.eval_file = argv[++i];
        } else if (a == "--train" && need(1)) {
            opt.train_file = argv[++i];
        } else if (a == "--batch_nodes" && need(1)) {
            opt.batch_nodes = std::stoul(argv[++i]);
        } else if (a == "--seed" && need(1)) {
            opt.seed = std::stoull(argv[++i]);
        } else {
            return false;
        }
    }
    return !opt.checkpoint.empty() && !opt.eval_file.empty();
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

static void add_truths(const MMapArray<Triple>& triples,
                       std::unordered_map<uint64_t, std::unordered_set<uint32_t>>& truth) {
    for (size_t i = 0; i < triples.size; ++i) {
        const Triple& t = triples[i];
        uint64_t key = (static_cast<uint64_t>(t.h) << 32) | t.r;
        truth[key].insert(t.t);
    }
}

int main(int argc, char** argv) {
    EvalOptions opt;
    if (!parse_args(argc, argv, opt)) {
        std::cerr << "Usage: kg_eval --checkpoint ckpt --eval eval.bin [--train train.bin] [--data dir] [--reverse dir]\n";
        return 1;
    }

    CsrGraph g(opt.data_dir);
    if (!g.valid()) {
        std::cerr << "Failed to load graph\n";
        return 1;
    }
    CsrGraph* rev_ptr = nullptr;
    CsrGraph rev;
    if (!opt.reverse_dir.empty()) {
        if (rev.load_custom(opt.reverse_dir, "offsets_rev.bin", "csr_rev.bin", "rels_rev.bin")) {
            rev_ptr = &rev;
        }
    }

    MMapArray<Triple> eval_q;
    if (!map_triples(opt.eval_file, eval_q)) {
        std::cerr << "Failed to load eval queries\n";
        return 1;
    }
    MMapArray<Triple> train_q;
    if (!opt.train_file.empty()) {
        map_triples(opt.train_file, train_q);
    }

    CheckpointMeta meta;
    std::vector<std::vector<float>> params, m, v;
    if (!load_checkpoint(opt.checkpoint, meta, params, m, v)) {
        std::cerr << "Failed to load checkpoint\n";
        return 1;
    }
    FeatureConfig fcfg = meta.feat_cfg;
    if (meta.enc_cfg.fanouts.empty()) meta.enc_cfg.fanouts.resize(meta.enc_cfg.layers, 10);

    XorShift128Plus rng(opt.seed);
    Encoder encoder(meta.feature_dim ? meta.feature_dim : feature_dim(fcfg, rev_ptr != nullptr),
                    g.num_relations(), meta.enc_cfg, fcfg, rng);
    Decoder decoder(g.num_relations(), meta.enc_cfg.hidden_dim, encoder.relation_embeddings(), rng);
    auto enc_params = encoder.parameters();
    auto dec_params = decoder.parameters();
    std::vector<Parameter*> all_params = enc_params;
    all_params.insert(all_params.end(), dec_params.begin(), dec_params.end());
    assign_parameters(params, all_params);

    std::vector<float> cache = build_cache(encoder, g, rev_ptr, encoder.output_dim(), opt.batch_nodes, rng);
    const float* rel_emb = encoder.relation_embeddings()->data.data();

    std::unordered_map<uint64_t, std::unordered_set<uint32_t>> truth;
    add_truths(eval_q, truth);
    if (train_q.data) add_truths(train_q, truth);

    Metrics metrics;
    size_t dim = encoder.output_dim();
    for (size_t i = 0; i < eval_q.size; ++i) {
        const Triple& q = eval_q[i];
        if (q.h == 0 || q.r == 0 || q.t == 0) continue;
        const float* h = &cache[(static_cast<size_t>(q.h - 1)) * dim];
        const float* t_true = &cache[(static_cast<size_t>(q.t - 1)) * dim];
        const float* rvec = &rel_emb[static_cast<size_t>(q.r) * dim];
        float true_score = 0.0f;
        for (size_t d = 0; d < dim; ++d) true_score += h[d] * rvec[d] * t_true[d];

        size_t rank = 1;
        for (uint32_t cand = 1; cand <= g.num_nodes(); ++cand) {
            if (cand == q.t) continue;
            uint64_t key = (static_cast<uint64_t>(q.h) << 32) | q.r;
            auto it = truth.find(key);
            if (it != truth.end() && it->second.count(cand)) continue;
            const float* t = &cache[(cand - 1) * dim];
            float score = 0.0f;
            for (size_t d = 0; d < dim; ++d) score += h[d] * rvec[d] * t[d];
            if (score > true_score) ++rank;
        }
        accumulate_rank(metrics, rank);
    }
    finalize_metrics(metrics);

    std::cout << "MRR=" << metrics.mrr << " Hits@1=" << metrics.hits1
              << " Hits@3=" << metrics.hits3 << " Hits@10=" << metrics.hits10
              << " Hits@100=" << metrics.hits100 << " (count=" << metrics.count << ")\n";
    return 0;
}
