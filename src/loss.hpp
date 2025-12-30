#pragma once

#include <cmath>

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline float logistic_loss(float score, int label) {
    float y = static_cast<float>(label);
    return std::log1pf(std::exp(-y * score));
}

