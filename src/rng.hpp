#pragma once

#include <cstdint>

class XorShift128Plus {
public:
    XorShift128Plus(uint64_t seed = 1, uint64_t seq = 0);
    uint64_t next_u64();
    uint32_t next_u32(uint32_t bound);
    float uniform();

private:
    uint64_t s0_;
    uint64_t s1_;
};

uint64_t mix_seed(uint64_t x);

