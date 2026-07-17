#pragma once

#include <cstddef>

namespace terlml {

struct rmsnorm_shape {
    std::size_t rows;
    std::size_t hidden;
};

void rmsnorm_f32(
    const float * input,
    const float * weight,
    float epsilon,
    float * output,
    rmsnorm_shape shape
);

} // namespace terlml
