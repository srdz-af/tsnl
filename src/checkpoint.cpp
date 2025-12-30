#include "checkpoint.hpp"

#include <cstring>
#include <fstream>
#include <iostream>

static void write_vec(std::ofstream& f, const std::vector<float>& v) {
    uint64_t n = v.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    if (n) f.write(reinterpret_cast<const char*>(v.data()), sizeof(float) * n);
}

static bool read_vec(std::ifstream& f, std::vector<float>& v) {
    uint64_t n = 0;
    if (!f.read(reinterpret_cast<char*>(&n), sizeof(n))) return false;
    v.resize(n);
    if (n) {
        if (!f.read(reinterpret_cast<char*>(v.data()), sizeof(float) * n)) return false;
    }
    return true;
}

bool save_checkpoint(const std::string& path, const Encoder& enc, const Decoder& dec,
                     const FeatureConfig& feat_cfg, const Optimizer* opt) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        std::cerr << "Failed to open checkpoint for write: " << path << "\n";
        return false;
    }
    uint32_t magic = 0x4b474331; // "KGC1"
    uint32_t version = 1;
    f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    f.write(reinterpret_cast<const char*>(&version), sizeof(version));

    const EncoderConfig& ec = enc.config();
    // Serialize encoder config
    uint64_t hidden = enc.output_dim();
    uint32_t layers = static_cast<uint32_t>(ec.layers);
    f.write(reinterpret_cast<const char*>(&hidden), sizeof(hidden));
    f.write(reinterpret_cast<const char*>(&layers), sizeof(layers));
    uint64_t fanout_len = ec.fanouts.size();
    f.write(reinterpret_cast<const char*>(&fanout_len), sizeof(fanout_len));
    for (size_t i = 0; i < ec.fanouts.size(); ++i) {
        uint64_t v = static_cast<uint64_t>(ec.fanouts[i]);
        f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }
    uint8_t use_relu = ec.use_relu ? 1 : 0;
    f.write(reinterpret_cast<const char*>(&use_relu), sizeof(use_relu));

    uint8_t use_in = feat_cfg.use_in_degree ? 1 : 0;
    uint8_t add_noise = feat_cfg.add_noise ? 1 : 0;
    f.write(reinterpret_cast<const char*>(&use_in), sizeof(use_in));
    f.write(reinterpret_cast<const char*>(&add_noise), sizeof(add_noise));

    uint64_t num_rel = enc.num_relations();
    f.write(reinterpret_cast<const char*>(&num_rel), sizeof(num_rel));
    uint64_t feature_dim = enc.feature_dim_raw();
    f.write(reinterpret_cast<const char*>(&feature_dim), sizeof(feature_dim));

    // Parameters
    auto params_enc = enc.parameters_const();
    auto params_dec = dec.parameters_const();
    std::vector<const Parameter*> params;
    params.insert(params.end(), params_enc.begin(), params_enc.end());
    params.insert(params.end(), params_dec.begin(), params_dec.end());

    uint32_t param_count = static_cast<uint32_t>(params.size());
    f.write(reinterpret_cast<const char*>(&param_count), sizeof(param_count));
    for (const auto* p : params) {
        write_vec(f, p->data);
    }

    bool use_adam = opt && !opt->m().empty();
    uint8_t use_adam_u8 = use_adam ? 1 : 0;
    f.write(reinterpret_cast<const char*>(&use_adam_u8), sizeof(use_adam_u8));
    uint64_t step = opt ? static_cast<uint64_t>(opt->step_count()) : 0;
    f.write(reinterpret_cast<const char*>(&step), sizeof(step));
    if (use_adam) {
        const auto& m = opt->m();
        const auto& v = opt->v();
        for (const auto& vec : m) write_vec(f, vec);
        for (const auto& vec : v) write_vec(f, vec);
    }

    return true;
}

bool load_checkpoint(const std::string& path, CheckpointMeta& meta,
                     std::vector<std::vector<float>>& params_out,
                     std::vector<std::vector<float>>& m_out,
                     std::vector<std::vector<float>>& v_out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open checkpoint for read: " << path << "\n";
        return false;
    }
    uint32_t magic = 0, version = 0;
    if (!f.read(reinterpret_cast<char*>(&magic), sizeof(magic))) return false;
    if (!f.read(reinterpret_cast<char*>(&version), sizeof(version))) return false;
    if (magic != 0x4b474331) {
        std::cerr << "Bad checkpoint magic\n";
        return false;
    }
    uint64_t hidden = 0;
    uint32_t layers = 0;
    uint64_t fanout_len = 0;
    if (!f.read(reinterpret_cast<char*>(&hidden), sizeof(hidden))) return false;
    if (!f.read(reinterpret_cast<char*>(&layers), sizeof(layers))) return false;
    if (!f.read(reinterpret_cast<char*>(&fanout_len), sizeof(fanout_len))) return false;
    meta.enc_cfg.hidden_dim = static_cast<size_t>(hidden);
    meta.enc_cfg.layers = static_cast<int>(layers);
    meta.enc_cfg.fanouts.assign(fanout_len, 0);
    for (uint64_t i = 0; i < fanout_len; ++i) {
        uint64_t v = 0;
        if (!f.read(reinterpret_cast<char*>(&v), sizeof(v))) return false;
        meta.enc_cfg.fanouts[i] = static_cast<size_t>(v);
    }
    uint8_t use_relu = 1;
    if (!f.read(reinterpret_cast<char*>(&use_relu), sizeof(use_relu))) return false;
    meta.enc_cfg.use_relu = (use_relu != 0);

    uint8_t use_in = 0, add_noise = 0;
    if (!f.read(reinterpret_cast<char*>(&use_in), sizeof(use_in))) return false;
    if (!f.read(reinterpret_cast<char*>(&add_noise), sizeof(add_noise))) return false;
    meta.feat_cfg.use_in_degree = (use_in != 0);
    meta.feat_cfg.add_noise = (add_noise != 0);

    uint64_t num_rel = 0, feature_dim = 0;
    if (!f.read(reinterpret_cast<char*>(&num_rel), sizeof(num_rel))) return false;
    if (!f.read(reinterpret_cast<char*>(&feature_dim), sizeof(feature_dim))) return false;
    meta.num_rel = static_cast<size_t>(num_rel);
    meta.feature_dim = static_cast<size_t>(feature_dim);

    uint32_t param_count = 0;
    if (!f.read(reinterpret_cast<char*>(&param_count), sizeof(param_count))) return false;
    params_out.resize(param_count);
    for (uint32_t i = 0; i < param_count; ++i) {
        if (!read_vec(f, params_out[i])) return false;
    }

    uint8_t use_adam_u8 = 0;
    if (!f.read(reinterpret_cast<char*>(&use_adam_u8), sizeof(use_adam_u8))) return false;
    meta.use_adam = use_adam_u8 != 0;
    uint64_t step = 0;
    if (!f.read(reinterpret_cast<char*>(&step), sizeof(step))) return false;
    meta.step = static_cast<size_t>(step);
    if (meta.use_adam) {
        m_out.resize(param_count);
        v_out.resize(param_count);
        for (uint32_t i = 0; i < param_count; ++i) if (!read_vec(f, m_out[i])) return false;
        for (uint32_t i = 0; i < param_count; ++i) if (!read_vec(f, v_out[i])) return false;
    }

    return true;
}

void assign_parameters(const std::vector<std::vector<float>>& data,
                       const std::vector<Parameter*>& params) {
    if (data.size() != params.size()) return;
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i].size() != params[i]->size()) continue;
        params[i]->data = data[i];
        params[i]->grad.assign(params[i]->size(), 0.0f);
    }
}
