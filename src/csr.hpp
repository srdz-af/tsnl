#pragma once

#include "io.hpp"

#include <cstdint>
#include <string>

struct AdjView {
    const uint32_t* dst = nullptr;
    const uint16_t* rel = nullptr;
    uint32_t size = 0;
};

class CsrGraph {
public:
    CsrGraph() = default;
    explicit CsrGraph(const std::string& dir) { load(dir); }
    ~CsrGraph();

    bool load(const std::string& dir);
    bool load_custom(const std::string& dir,
                     const std::string& offsets_file,
                     const std::string& csr_file,
                     const std::string& rels_file,
                     const std::string& entities_file = "entities.bin",
                     const std::string& props_file = "props.bin");
    uint32_t num_nodes() const { return n_; }
    uint32_t num_edges() const { return m_; }
    uint32_t num_relations() const { return r_; }
    AdjView neighbors(uint32_t v) const;
    uint32_t out_degree(uint32_t v) const;
    uint32_t entity_of(uint32_t v) const;
    uint16_t prop_of(uint32_t r) const;
    bool valid() const { return n_ > 0; }

private:
    MMapArray<uint32_t> offsets_;
    MMapArray<uint32_t> csr_;
    MMapArray<uint16_t> rels_;
    MMapArray<uint32_t> entities_;
    MMapArray<uint16_t> props_;
    uint32_t n_ = 0;
    uint32_t m_ = 0;
    uint32_t r_ = 0;
};
