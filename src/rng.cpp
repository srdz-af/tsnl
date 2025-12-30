#include "rng.hpp"

XorShift128Plus::XorShift128Plus(uint64_t seed, uint64_t seq) {
    s0_ = mix_seed(seed);
    s1_ = mix_seed(seq + 0x9e3779b97f4a7c15ULL);
    if ((s0_ | s1_) == 0) s1_ = 1;
}

uint64_t XorShift128Plus::next_u64() {
    uint64_t x = s0_;
    uint64_t y = s1_;
    s0_ = y;
    x ^= x << 23;
    s1_ = x ^ y ^ (x >> 17) ^ (y >> 26);
    return s1_ + y;
}

uint32_t XorShift128Plus::next_u32(uint32_t bound) {
    uint64_t r = next_u64();
    return static_cast<uint32_t>(r % bound);
}

float XorShift128Plus::uniform() {
    return (next_u64() >> 11) * (1.0f / 9007199254740992.0f);
}

uint64_t mix_seed(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
