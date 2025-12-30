#pragma once

#include <cstddef>

struct Metrics {
    double mrr = 0.0;
    double hits1 = 0.0;
    double hits3 = 0.0;
    double hits10 = 0.0;
    double hits100 = 0.0;
    size_t count = 0;
};

void accumulate_rank(Metrics& m, size_t rank);
void finalize_metrics(Metrics& m);
