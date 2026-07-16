#pragma once

#include <cstddef>

namespace terlml {

struct hgrn_attention_block_shape {
    std::size_t batch;
    std::size_t sequence;
    std::size_t hidden;
    std::size_t heads;
    std::size_t conv_kernel_size;
};

struct hgrn_attention_block_weights {
    const float * conv_weight;
    const float * conv_bias;

    const float * i_proj_weight;
    const float * f_proj_weight;
    const float * g_proj_weight;

    const float * gnorm_weight;
    const float * o_proj_weight;
};

struct hgrn_attention_block_state {
    const float * conv_cache;
    const float * recurrent_state;
};

struct hgrn_attention_block_output_state {
    float * conv_cache;
    float * recurrent_state;
};

void hgrn_attention_block_f32(
    const float * input,
    const hgrn_attention_block_weights & weights,
    hgrn_attention_block_state initial_state,
    float * output,
    hgrn_attention_block_output_state final_state,
    float rmsnorm_epsilon,
    hgrn_attention_block_shape shape
);

} // namespace terlml
