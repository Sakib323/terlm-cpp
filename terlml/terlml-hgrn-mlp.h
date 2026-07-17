#pragma once

#include <cstddef>

namespace terlml {

struct hgrn_mlp_shape {
    std::size_t batch;
    std::size_t sequence;
    std::size_t hidden;
    std::size_t intermediate;
};

struct hgrn_mlp_weights {
    const float * gate_proj_norm_weight;
    const float * gate_proj_weight;

    const float * down_proj_norm_weight;
    const float * down_proj_weight;
};

void hgrn_mlp_f32(
    const float * input,
    const hgrn_mlp_weights & weights,
    float * output,
    hgrn_mlp_shape shape
);

} // namespace terlml
