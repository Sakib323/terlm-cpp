#pragma once

#include "terlml-hgrn-attention-block.h"
#include "terlml-hgrn-mlp.h"

#include <cstddef>

namespace terlml {

struct hgrn_decoder_layer_shape {
    std::size_t batch;
    std::size_t sequence;
    std::size_t hidden;
    std::size_t heads;
    std::size_t intermediate;
    std::size_t conv_kernel_size;
};

struct hgrn_decoder_layer_weights {
    hgrn_attention_block_weights attention;
    hgrn_mlp_weights mlp;
};

void hgrn_decoder_layer_f32(
    const float * input,
    const hgrn_decoder_layer_weights & weights,
    hgrn_attention_block_state initial_state,
    float * output,
    hgrn_attention_block_output_state final_state,
    float rmsnorm_epsilon,
    hgrn_decoder_layer_shape shape,
    const float * lower_bound = nullptr,
    float * attention_output_trace = nullptr
);

} // namespace terlml
