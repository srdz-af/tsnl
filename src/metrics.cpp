#include "metrics.hpp"

void accumulate_rank(Metrics& m, size_t rank) {
    ++m.count;
    m.mrr += 1.0 / static_cast<double>(rank);
    if (rank <= 1) m.hits1 += 1.0;
    if (rank <= 3) m.hits3 += 1.0;
    if (rank <= 10) m.hits10 += 1.0;
    if (rank <= 100) m.hits100 += 1.0;
}

void finalize_metrics(Metrics& m) {
    if (m.count == 0) return;
    double inv = 1.0 / static_cast<double>(m.count);
    m.mrr *= inv;
    m.hits1 *= inv;
    m.hits3 *= inv;
    m.hits10 *= inv;
    m.hits100 *= inv;
}
