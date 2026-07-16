#pragma once

#include <cstddef>

namespace terlml {

struct rmsnorm_gate_shape {
    std::size_t batch;
    std::size_t sequence;
    std::size_t hidden;
};

void rmsnorm_silu_gate_f32(
    const float * gate,
    const float * input,
    const float * weight,
    float epsilon,
    float * output,
    rmsnorm_gate_shape shape
);

} // namespace terlml
