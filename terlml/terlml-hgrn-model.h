#pragma once

#include "terlml-hgrn-decoder-layer.h"

#include <cstddef>
#include <cstdint>

namespace terlml {

struct hgrn_model_shape {
    std::size_t batch;
    std::size_t sequence;
    std::size_t vocab;
    std::size_t layers;
    std::size_t hidden;
    std::size_t heads;
    std::size_t intermediate;
    std::size_t conv_kernel_size;
};

struct hgrn_model_weights {
    const float * embedding_weight;
    const hgrn_decoder_layer_weights * decoder_layers;
    const float * raw_lower_bounds;
    const float * final_norm_weight;
    const float * lm_head_norm_weight;
    const float * lm_head_weight;
};

struct hgrn_model_state {
    const float * conv_cache;
    const float * recurrent_state;
};

struct hgrn_model_output_state {
    float * conv_cache;
    float * recurrent_state;
};

using hgrn_model_layer_trace_callback = void (*)(
    void * context,
    std::size_t layer,
    const float * attention_output,
    const float * conv_cache,
    const float * recurrent_state,
    hgrn_model_shape shape
);

void hgrn_model_f32(
    const std::uint32_t * token_ids,
    const hgrn_model_weights & weights,
    hgrn_model_state initial_state,
    float * logits,
    hgrn_model_output_state final_state,
    float rmsnorm_epsilon,
    hgrn_model_shape shape,
    hgrn_model_layer_trace_callback trace_callback = nullptr,
    void * trace_context = nullptr
);

} // namespace terlml
