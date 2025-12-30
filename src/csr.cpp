#include "csr.hpp"

#include <iostream>
#include <sstream>

CsrGraph::~CsrGraph() {
    unmap(offsets_.base);
    unmap(csr_.base);
    unmap(rels_.base);
    unmap(entities_.base);
    unmap(props_.base);
}

bool CsrGraph::load(const std::string& dir) {
    return load_custom(dir, "offsets.bin", "csr.bin", "rels.bin", "entities.bin", "props.bin");
}

bool CsrGraph::load_custom(const std::string& dir,
                           const std::string& offsets_file,
                           const std::string& csr_file,
                           const std::string& rels_file,
                           const std::string& entities_file,
                           const std::string& props_file) {
    std::string offsets_path  = dir + "/" + offsets_file;
    std::string csr_path      = dir + "/" + csr_file;
    std::string rels_path     = dir + "/" + rels_file;
    std::string entities_path = dir + "/" + entities_file;
    std::string props_path    = dir + "/" + props_file;

    if (!map_array(offsets_path, offsets_)) return false;
    if (offsets_.size < 2) return false;
    n_ = static_cast<uint32_t>(offsets_.size - 2);
    m_ = offsets_[n_ + 1];

    if (!map_array(csr_path, csr_)) return false;
    if (!map_array(rels_path, rels_)) return false;
    if (csr_.size != m_ || rels_.size != m_) {
        std::cerr << "CSR size mismatch\n";
        return false;
    }

    if (!map_array(entities_path, entities_)) return false;
    if (entities_.size != n_) {
        std::cerr << "Entity map length mismatch\n";
        return false;
    }

    if (!map_array(props_path, props_)) return false;
    r_ = static_cast<uint32_t>(props_.size);

    return true;
}

AdjView CsrGraph::neighbors(uint32_t v) const {
    if (v == 0 || v > n_) return {};
    uint32_t begin = offsets_[v];
    uint32_t end   = offsets_[v + 1];
    AdjView a;
    a.dst  = csr_.data + begin;
    a.rel  = rels_.data + begin;
    a.size = end - begin;
    return a;
}

uint32_t CsrGraph::out_degree(uint32_t v) const {
    if (v == 0 || v > n_) return 0;
    return offsets_[v + 1] - offsets_[v];
}

uint32_t CsrGraph::entity_of(uint32_t v) const {
    if (v == 0 || v > n_) return 0;
    return entities_[v - 1];
}

uint16_t CsrGraph::prop_of(uint32_t r) const {
    if (r == 0 || r > r_) return 0;
    return props_[r - 1];
}
