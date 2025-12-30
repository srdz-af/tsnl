#include "csr.hpp"
#include "decoder.hpp"
#include "encoder.hpp"
#include "features.hpp"
#include "io.hpp"
#include "optim.hpp"
#include "rng.hpp"

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

static std::string make_temp_dir() {
    std::string templ = "/tmp/kgtestXXXXXX";
    std::vector<char> buf(templ.begin(), templ.end());
    buf.push_back('\0');
    char* path = mkdtemp(buf.data());
    return path ? std::string(path) : std::string();
}

int main() {
    std::string dir = make_temp_dir();
    assert(!dir.empty());

    // Tiny graph: 3 nodes, 3 edges forming a chain
    std::vector<uint32_t> offsets = {0, 1, 2, 3, 3};
    std::vector<uint32_t> csr = {2, 3, 1};
    std::vector<uint16_t> rels = {1, 1, 1};
    std::vector<uint32_t> entities = {100, 200, 300};
    std::vector<uint16_t> props = {10};

    assert(write_array(dir + "/offsets.bin", offsets));
    assert(write_array(dir + "/csr.bin", csr));
    assert(write_array(dir + "/rels.bin", rels));
    assert(write_array(dir + "/entities.bin", entities));
    assert(write_array(dir + "/props.bin", props));

    CsrGraph g(dir);
    assert(g.valid());
    assert(g.num_nodes() == 3);
    assert(g.num_edges() == 3);

    FeatureConfig fcfg;
    fcfg.use_in_degree = false;
    size_t feat_dim = feature_dim(fcfg, false);

    EncoderConfig ecfg;
    ecfg.hidden_dim = 8;
    ecfg.layers = 1;
    ecfg.fanouts = {1};

    XorShift128Plus rng_init(1);
    Encoder enc(feat_dim, g.num_relations(), ecfg, fcfg, rng_init);
    Decoder dec(g.num_relations(), ecfg.hidden_dim, enc.relation_embeddings(), rng_init);

    std::vector<uint32_t> heads = {1};
    std::vector<uint32_t> rel_ids = {1};
    std::vector<uint32_t> tails = {2};
    std::vector<uint32_t> negs = {3};

    std::vector<uint32_t> seeds = {1, 2, 3};
    XorShift128Plus rng1(2);
    EncoderState st1 = enc.forward(g, nullptr, seeds, rng1);
    std::vector<std::vector<float>> grad1(ecfg.fanouts.size() + 1);
    grad1.back().assign(st1.sg.nodes_per_layer.back().size() * ecfg.hidden_dim, 0.0f);
    float loss1 = dec.distmult_loss(heads, rel_ids, tails, negs, 1,
                                    st1.index_per_layer.back(), st1.h_layers.back(),
                                    grad1.back());
    dec.relation_loss(heads, tails, rel_ids, st1.index_per_layer.back(), st1.h_layers.back(),
                      grad1.back());
    enc.backward(st1, grad1);

    std::vector<Parameter*> params = enc.parameters();
    auto dp = dec.parameters();
    params.insert(params.end(), dp.begin(), dp.end());
    OptimConfig oc;
    oc.lr = 0.01f;
    oc.use_adam = false;
    Optimizer opt(oc, params);
    opt.step();

    // Re-run after one step; loss should not increase dramatically.
    XorShift128Plus rng2(2);
    EncoderState st2 = enc.forward(g, nullptr, seeds, rng2);
    std::vector<std::vector<float>> grad2(ecfg.fanouts.size() + 1);
    grad2.back().assign(st2.sg.nodes_per_layer.back().size() * ecfg.hidden_dim, 0.0f);
    float loss2 = dec.distmult_loss(heads, rel_ids, tails, negs, 1,
                                    st2.index_per_layer.back(), st2.h_layers.back(),
                                    grad2.back());

    assert(loss2 < loss1 + 1e-3f);
    fs::remove_all(dir);
    std::printf("sanity ok (loss %.4f -> %.4f)\n", loss1, loss2);
    return 0;
}
