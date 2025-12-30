#pragma once

#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

inline size_t default_threads() {
    size_t n = std::thread::hardware_concurrency();
    return n == 0 ? 4 : n;
}

template <typename Fn>
void parallel_for(size_t begin, size_t end, size_t num_threads, Fn fn) {
    if (num_threads <= 1 || end <= begin + 1) {
        for (size_t i = begin; i < end; ++i) fn(i);
        return;
    }
    size_t total = end - begin;
    size_t chunk = (total + num_threads - 1) / num_threads;
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    size_t cur = begin;
    for (size_t t = 0; t < num_threads && cur < end; ++t) {
        size_t start = cur;
        size_t stop = std::min(end, start + chunk);
        workers.emplace_back([start, stop, &fn]() {
            for (size_t i = start; i < stop; ++i) fn(i);
        });
        cur = stop;
    }
    for (auto& th : workers) th.join();
}

