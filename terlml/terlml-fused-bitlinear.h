#pragma once

#include <cstddef>

namespace terlml {

struct fused_bitlinear_shape {
    std::size_t rows;
    std::size_t input_features;
    std::size_t output_features;
};

void fused_bitlinear_f32(
    const float * input,
    const float * norm_weight,
    const float * linear_weight,
    float * output,
    float epsilon,
    fused_bitlinear_shape shape
);

} // namespace terlml
