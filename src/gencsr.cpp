#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;

static constexpr uint32_t MAXQ = 137371733u;
static constexpr uint32_t MAXP = 13986u;

static inline const char* skip_ws(const char* p, const char* end) {
    while (p < end) {
        char c = *p;
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { ++p; continue; }
        break;
    }
    return p;
}

static inline bool parse_u32(const char*& p, const char* end, uint32_t& out) {
    if (p >= end) return false;
    uint64_t v = 0;
    const char* start = p;
    while (p < end) {
        char c = *p;
        if (c < '0' || c > '9') break;
        v = v * 10 + (uint64_t)(c - '0');
        if (v > std::numeric_limits<uint32_t>::max()) return false;
        ++p;
    }
    if (p == start) return false;
    out = (uint32_t)v;
    return true;
}

static inline bool parse_qpq(const char* s, size_t n, uint32_t& qs, uint32_t& pp, uint32_t& qo) {
    static constexpr char PRE_S[] = "<http://www.wikidata.org/entity/Q";
    static constexpr char PRE_P[] = "<http://www.wikidata.org/prop/direct/P";

    const char* p = s;
    const char* end = s + n;

    if ((size_t)(end - p) < sizeof(PRE_S) - 1) return false;
    if (memcmp(p, PRE_S, sizeof(PRE_S) - 1) != 0) return false;
    p += sizeof(PRE_S) - 1;
    if (!parse_u32(p, end, qs)) return false;
    if (p >= end || *p != '>') return false;
    ++p;

    p = skip_ws(p, end);

    if ((size_t)(end - p) < sizeof(PRE_P) - 1) return false;
    if (memcmp(p, PRE_P, sizeof(PRE_P) - 1) != 0) return false;
    p += sizeof(PRE_P) - 1;
    if (!parse_u32(p, end, pp)) return false;
    if (p >= end || *p != '>') return false;
    ++p;

    p = skip_ws(p, end);

    if ((size_t)(end - p) < sizeof(PRE_S) - 1) return false;
    if (memcmp(p, PRE_S, sizeof(PRE_S) - 1) != 0) return false;
    p += sizeof(PRE_S) - 1;
    if (!parse_u32(p, end, qo)) return false;
    if (p >= end || *p != '>') return false;
    ++p;

    p = skip_ws(p, end);
    if (p >= end || *p != '.') return false;

    return true;
}

static void* map_file_rw(const char* path, size_t bytes, int& fd_out) {
    fd_out = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd_out < 0) return MAP_FAILED;
    if (ftruncate(fd_out, (off_t)bytes) != 0) {
        close(fd_out);
        return MAP_FAILED;
    }
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd_out, 0);
    if (p == MAP_FAILED) {
        close(fd_out);
        return MAP_FAILED;
    }
    return p;
}

#pragma pack(push, 1)
struct EdgeRec {
    uint32_t s;
    uint16_t p;
    uint32_t o;
};
#pragma pack(pop)
static_assert(sizeof(EdgeRec) == 10, "EdgeRec must be 10 bytes");

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    mkdir("data", 0755);

    uint32_t* idx_q = (uint32_t*)calloc((size_t)MAXQ + 2, sizeof(uint32_t));
    uint16_t* idx_p = (uint16_t*)calloc((size_t)MAXP + 2, sizeof(uint16_t));
    if (!idx_q || !idx_p) return 1;

    FILE* f_entities = fopen("data/entities.bin", "wb");
    FILE* f_props    = fopen("data/props.bin", "wb");
    FILE* f_edges    = fopen("data/edges.tmp", "w+b");
    if (!f_entities || !f_props || !f_edges) return 1;

    static char ent_buf[1 << 20];
    static char prop_buf[1 << 16];
    static char edge_buf_io[1 << 20];
    setvbuf(f_entities, ent_buf, _IOFBF, sizeof(ent_buf));
    setvbuf(f_props,    prop_buf, _IOFBF, sizeof(prop_buf));
    setvbuf(f_edges,    edge_buf_io, _IOFBF, sizeof(edge_buf_io));

    vector<uint32_t> deg;
    deg.reserve(1 << 20);
    deg.push_back(0); 

    uint32_t iq = 1;    
    uint16_t ip = 1;    
    uint64_t m  = 0;    

    constexpr size_t BLOCK = 1 << 20;
    vector<EdgeRec> edge_block;
    edge_block.reserve(BLOCK);

    constexpr size_t ENT_BLOCK = 1 << 20;
    vector<uint32_t> ent_block;
    ent_block.reserve(ENT_BLOCK);

    string line;
    while (getline(cin, line)) {
        uint32_t s_orig, p_orig, o_orig;
        if (!parse_qpq(line.data(), line.size(), s_orig, p_orig, o_orig)) continue;
        if (s_orig > MAXQ || o_orig > MAXQ || p_orig > MAXP) continue;

        uint32_t ds = idx_q[s_orig];
        if (!ds) {
            ds = iq++;
            idx_q[s_orig] = ds;
            deg.push_back(0);
            ent_block.push_back(s_orig);
            if (ent_block.size() == ENT_BLOCK) {
                fwrite(ent_block.data(), sizeof(uint32_t), ent_block.size(), f_entities);
                ent_block.clear();
            }
        }

        uint16_t dp = idx_p[p_orig];
        if (!dp) {
            dp = ip++;
            idx_p[p_orig] = dp;
            uint16_t porig16 = (uint16_t)p_orig;
            fwrite(&porig16, sizeof(uint16_t), 1, f_props);
        }

        uint32_t do_ = idx_q[o_orig];
        if (!do_) {
            do_ = iq++;
            idx_q[o_orig] = do_;
            deg.push_back(0);
            ent_block.push_back(o_orig);
            if (ent_block.size() == ENT_BLOCK) {
                fwrite(ent_block.data(), sizeof(uint32_t), ent_block.size(), f_entities);
                ent_block.clear();
            }
        }

        deg[ds]++;
        ++m;

        edge_block.push_back(EdgeRec{ds, dp, do_});
        if (edge_block.size() == BLOCK) {
            fwrite(edge_block.data(), sizeof(EdgeRec), edge_block.size(), f_edges);
            edge_block.clear();
        }
    }

    if (!ent_block.empty()) fwrite(ent_block.data(), sizeof(uint32_t), ent_block.size(), f_entities);
    if (!edge_block.empty()) fwrite(edge_block.data(), sizeof(EdgeRec), edge_block.size(), f_edges);

    fflush(f_entities);
    fflush(f_props);
    fflush(f_edges);
    fclose(f_entities);
    fclose(f_props);

    if (m > std::numeric_limits<uint32_t>::max()) return 1;

    uint64_t sum = 0;
    for (size_t u = 0; u < deg.size(); ++u) {
        uint32_t cnt = deg[u];
        deg[u] = (uint32_t)sum;
        sum += cnt;
    }
    if (sum != m) return 1;

    FILE* f_offsets = fopen("data/offsets.bin", "wb");
    if (!f_offsets) return 1;
    static char off_buf[1 << 20];
    setvbuf(f_offsets, off_buf, _IOFBF, sizeof(off_buf));

    fwrite(deg.data(), sizeof(uint32_t), deg.size(), f_offsets);
    uint32_t final_off = (uint32_t)sum;
    fwrite(&final_off, sizeof(uint32_t), 1, f_offsets);
    fclose(f_offsets);

    free(idx_q);
    free(idx_p);

    const size_t csr_bytes  = (size_t)m * sizeof(uint32_t);
    const size_t rels_bytes = (size_t)m * sizeof(uint16_t);

    int fd_csr = -1, fd_rels = -1;
    void* csr_map  = map_file_rw("data/csr.bin",  csr_bytes,  fd_csr);
    void* rels_map = map_file_rw("data/rels.bin", rels_bytes, fd_rels);
    if (csr_map == MAP_FAILED || rels_map == MAP_FAILED) return 1;

    auto* csr  = (uint32_t*)csr_map;
    auto* rels = (uint16_t*)rels_map;

    fseek(f_edges, 0, SEEK_SET);
    vector<EdgeRec> read_block(BLOCK);

    while (true) {
        size_t nread = fread(read_block.data(), sizeof(EdgeRec), BLOCK, f_edges);
        if (nread == 0) break;
        for (size_t i = 0; i < nread; ++i) {
            const EdgeRec& e = read_block[i];
            uint32_t pos = deg[e.s]++;
            csr[pos]  = e.o;
            rels[pos] = e.p;
        }
    }

    fclose(f_edges);
    unlink("data/edges.tmp");

    msync(csr_map, csr_bytes, MS_SYNC);
    msync(rels_map, rels_bytes, MS_SYNC);

    munmap(csr_map, csr_bytes);
    munmap(rels_map, rels_bytes);
    close(fd_csr);
    close(fd_rels);

    cerr << "nodes=" << (iq - 1) << " rels=" << (uint32_t)(ip - 1) << " edges=" << m << "\n";
    return 0;
}
