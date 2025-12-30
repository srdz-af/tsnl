#include "csr.hpp"
#include "io.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string in_dir = "data";
    std::string out_dir = "data";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--input" || arg == "-i") && i + 1 < argc) {
            in_dir = argv[++i];
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            out_dir = argv[++i];
        }
    }

    CsrGraph g(in_dir);
    if (!g.valid()) {
        std::cerr << "Failed to load graph from " << in_dir << "\n";
        return 1;
    }
    uint32_t n = g.num_nodes();
    uint32_t m = g.num_edges();
    std::vector<uint32_t> indeg(n + 1, 0);
    for (uint32_t u = 1; u <= n; ++u) {
        AdjView adj = g.neighbors(u);
        for (uint32_t i = 0; i < adj.size; ++i) {
            uint32_t v = adj.dst[i];
            if (v <= n) ++indeg[v];
        }
    }

    std::vector<uint32_t> offsets(n + 2, 0);
    uint64_t cur = 0;
    for (uint32_t v = 1; v <= n; ++v) {
        offsets[v] = static_cast<uint32_t>(cur);
        cur += indeg[v];
    }
    offsets[n + 1] = static_cast<uint32_t>(cur);

    std::vector<uint32_t> csr(m, 0);
    std::vector<uint16_t> rels(m, 0);
    std::vector<uint32_t> cursor(n + 1, 0);
    for (uint32_t v = 1; v <= n; ++v) cursor[v] = offsets[v];

    for (uint32_t u = 1; u <= n; ++u) {
        AdjView adj = g.neighbors(u);
        for (uint32_t i = 0; i < adj.size; ++i) {
            uint32_t v = adj.dst[i];
            uint16_t r = adj.rel[i];
            uint32_t pos = cursor[v]++;
            csr[pos] = u;
            rels[pos] = r;
        }
    }

    if (!write_array(out_dir + "/offsets_rev.bin", offsets)) return 1;
    if (!write_array(out_dir + "/csr_rev.bin", csr)) return 1;
    if (!write_array(out_dir + "/rels_rev.bin", rels)) return 1;

    std::cout << "Reverse CSR written to " << out_dir << " (nodes=" << n << " edges=" << m << ")\n";
    return 0;
}
