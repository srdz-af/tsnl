#include "checkpoint.hpp"
#include "csr.hpp"
#include "decoder.hpp"
#include "encoder.hpp"
#include "io.hpp"
#include "optim.hpp"
#include "rng.hpp"

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

struct TrainOptions {
    std::string data_dir = "data";
    std::string reverse_dir;
    std::string train_file;
    std::string checkpoint = "checkpoint.bin";
    size_t epochs = 1;
    size_t batch_size = 256;
    size_t dim = 64;
    int layers = 2;
    size_t fanout1 = 20;
    size_t fanout2 = 10;
    size_t negatives = 5;
    float lambda_rel = 1.0f;
    float lr = 0.001f;
    bool use_adam = true;
    bool use_in_degree = true;
    bool add_noise = false;
    uint64_t seed = 1;
};

static void print_usage() {
    std::cout << "Usage: kg_train --train train.bin [--data data_dir] [--reverse rev_dir] "
                 "[--epochs E] [--batch B] [--dim D] [--layers L] [--negatives K] "
                 "[--lambda_rel X] [--lr LR] [--optimizer adam|sgd] [--checkpoint path]\n";
}

static bool parse_args(int argc, char** argv, TrainOptions& opt) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](int more) { return i + more < argc; };
        if ((a == "--data" || a == "-d") && need(1)) {
            opt.data_dir = argv[++i];
        } else if (a == "--reverse" && need(1)) {
            opt.reverse_dir = argv[++i];
        } else if (a == "--train" && need(1)) {
            opt.train_file = argv[++i];
        } else if (a == "--epochs" && need(1)) {
            opt.epochs = std::stoul(argv[++i]);
        } else if (a == "--batch" && need(1)) {
            opt.batch_size = std::stoul(argv[++i]);
        } else if (a == "--dim" && need(1)) {
            opt.dim = std::stoul(argv[++i]);
        } else if (a == "--layers" && need(1)) {
            opt.layers = std::stoi(argv[++i]);
        } else if (a == "--fanout1" && need(1)) {
            opt.fanout1 = std::stoul(argv[++i]);
        } else if (a == "--fanout2" && need(1)) {
            opt.fanout2 = std::stoul(argv[++i]);
        } else if (a == "--negatives" && need(1)) {
            opt.negatives = std::stoul(argv[++i]);
        } else if (a == "--lambda_rel" && need(1)) {
            opt.lambda_rel = std::stof(argv[++i]);
        } else if (a == "--lr" && need(1)) {
            opt.lr = std::stof(argv[++i]);
        } else if (a == "--optimizer" && need(1)) {
            std::string v = argv[++i];
            opt.use_adam = (v != "sgd");
        } else if (a == "--checkpoint" && need(1)) {
            opt.checkpoint = argv[++i];
        } else if (a == "--no_in_degree") {
            opt.use_in_degree = false;
        } else if (a == "--noise") {
            opt.add_noise = true;
        } else if (a == "--seed" && need(1)) {
            opt.seed = std::stoull(argv[++i]);
        } else {
            print_usage();
            return false;
        }
    }
    if (opt.train_file.empty()) {
        print_usage();
        return false;
    }
    return true;
}

static void shuffle_indices(std::vector<size_t>& idx, XorShift128Plus& rng) {
    for (size_t i = idx.size(); i > 1; --i) {
        size_t j = rng.next_u32(static_cast<uint32_t>(i));
        std::swap(idx[i - 1], idx[j]);
    }
}

int main(int argc, char** argv) {
    TrainOptions opt;
    if (!parse_args(argc, argv, opt)) return 1;

    CsrGraph g(opt.data_dir);
    if (!g.valid()) {
        std::cerr << "Failed to load CSR from " << opt.data_dir << "\n";
        return 1;
    }
    CsrGraph* rev_ptr = nullptr;
    CsrGraph rev;
    if (!opt.reverse_dir.empty()) {
        if (rev.load_custom(opt.reverse_dir, "offsets_rev.bin", "csr_rev.bin", "rels_rev.bin")) {
            rev_ptr = &rev;
        } else {
            std::cerr << "Warning: reverse CSR could not be loaded; continuing without it.\n";
        }
    }

    MMapArray<Triple> train;
    if (!map_triples(opt.train_file, train)) {
        std::cerr << "Failed to map training triples.\n";
        return 1;
    }

    FeatureConfig fcfg;
    fcfg.use_in_degree = opt.use_in_degree && rev_ptr;
    fcfg.add_noise = opt.add_noise;
    size_t feat_dim = feature_dim(fcfg, rev_ptr != nullptr);

    EncoderConfig ecfg;
    ecfg.hidden_dim = opt.dim;
    ecfg.layers = opt.layers;
    ecfg.use_relu = true;
    ecfg.fanouts.clear();
    for (int l = 0; l < ecfg.layers; ++l) {
        if (l == 0)
            ecfg.fanouts.push_back(opt.fanout1);
        else
            ecfg.fanouts.push_back(opt.fanout2);
    }

    XorShift128Plus rng(opt.seed);
    Encoder encoder(feat_dim, g.num_relations(), ecfg, fcfg, rng);
    Decoder decoder(g.num_relations(), ecfg.hidden_dim, encoder.relation_embeddings(), rng);

    std::vector<Parameter*> params = encoder.parameters();
    auto dparams = decoder.parameters();
    params.insert(params.end(), dparams.begin(), dparams.end());
    OptimConfig ocfg;
    ocfg.lr = opt.lr;
    ocfg.use_adam = opt.use_adam;
    Optimizer optim(ocfg, params);

    size_t total = train.size;
    std::vector<size_t> order(total);
    for (size_t i = 0; i < total; ++i) order[i] = i;

    size_t neg_per = opt.negatives;
    size_t layer_L = ecfg.fanouts.size();

    for (size_t epoch = 0; epoch < opt.epochs; ++epoch) {
        shuffle_indices(order, rng);
        double epoch_loss = 0.0;
        size_t batches = 0;
        auto t0 = std::chrono::steady_clock::now();

        for (size_t start = 0; start < total; start += opt.batch_size) {
            size_t end = std::min(total, start + opt.batch_size);
            size_t bs = end - start;
            std::vector<uint32_t> heads(bs), rels(bs), tails(bs);
            for (size_t i = 0; i < bs; ++i) {
                const Triple& tr = train[order[start + i]];
                heads[i] = tr.h;
                rels[i] = tr.r;
                tails[i] = tr.t;
            }

            std::vector<uint32_t> neg_tails(bs * neg_per);
            for (size_t i = 0; i < bs * neg_per; ++i) {
                neg_tails[i] = sample_negative(g.num_nodes(), rng);
            }

            std::unordered_set<uint32_t> seed_set;
            seed_set.reserve(bs * (2 + neg_per) + 1);
            for (uint32_t v : heads) seed_set.insert(v);
            for (uint32_t v : tails) seed_set.insert(v);
            for (uint32_t v : neg_tails) seed_set.insert(v);
            std::vector<uint32_t> batch_nodes(seed_set.begin(), seed_set.end());

            optim.zero_grad();
            EncoderState st = encoder.forward(g, rev_ptr, batch_nodes, rng);
            std::vector<std::vector<float>> grad_layers(layer_L + 1);
            grad_layers[layer_L].assign(st.sg.nodes_per_layer[layer_L].size() * ecfg.hidden_dim, 0.0f);

            const auto& index_map = st.index_per_layer[layer_L];
            const auto& embeds = st.h_layers[layer_L];

            float loss_tail = decoder.distmult_loss(heads, rels, tails, neg_tails, neg_per,
                                                    index_map, embeds, grad_layers[layer_L]);
            float loss_rel = decoder.relation_loss(heads, tails, rels, index_map, embeds,
                                                   grad_layers[layer_L], opt.lambda_rel);

            encoder.backward(st, grad_layers);
            optim.step();

            epoch_loss += (loss_tail + loss_rel);
            ++batches;
        }

        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        double avg = batches ? (epoch_loss / batches) : 0.0;
        std::cout << "Epoch " << (epoch + 1) << "/" << opt.epochs
                  << " batches=" << batches << " loss=" << avg
                  << " time=" << dt << "s\n";
    }

    if (!save_checkpoint(opt.checkpoint, encoder, decoder, fcfg, &optim)) {
        std::cerr << "Failed to write checkpoint\n";
        return 1;
    }
    std::cout << "Checkpoint written to " << opt.checkpoint << "\n";
    return 0;
}
