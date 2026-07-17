#include "terlml-hgrn-model.h"

#include "terlml-fused-bitlinear.h"
#include "terlml-hgrn-lower-bounds.h"
#include "terlml-rmsnorm.h"

#include <cassert>
#include <algorithm>
#include <vector>

namespace terlml {

void hgrn_model_f32(
    const std::uint32_t * token_ids,
    const hgrn_model_weights & weights,
    hgrn_model_state initial_state,
    float * logits,
    hgrn_model_output_state final_state,
    float rmsnorm_epsilon,
    hgrn_model_shape shape
) {
    assert(token_ids != nullptr);
    assert(weights.embedding_weight != nullptr);
    assert(weights.decoder_layers != nullptr);
    assert(weights.raw_lower_bounds != nullptr);
    assert(weights.final_norm_weight != nullptr);
    assert(weights.lm_head_norm_weight != nullptr);
    assert(weights.lm_head_weight != nullptr);
    assert(initial_state.conv_cache != nullptr);
    assert(initial_state.recurrent_state != nullptr);
    assert(logits != nullptr);
    assert(final_state.conv_cache != nullptr);
    assert(final_state.recurrent_state != nullptr);

    assert(shape.batch > 0);
    assert(shape.sequence > 0);
    assert(shape.vocab > 0);
    assert(shape.layers > 0);
    assert(shape.hidden > 0);
    assert(shape.heads > 0);
    assert(shape.hidden % shape.heads == 0);
    assert(shape.intermediate > 0);
    assert(shape.conv_kernel_size > 0);

    const std::size_t rows = shape.batch * shape.sequence;
    const std::size_t hidden_elements = rows * shape.hidden;
    const std::size_t conv_state_per_layer =
        shape.batch * shape.hidden * shape.conv_kernel_size;
    const std::size_t recurrent_state_per_layer =
        shape.batch * shape.hidden;

    std::vector<float> current(hidden_elements, 0.0f);
    std::vector<float> next(hidden_elements, 0.0f);
    std::vector<float> lower_bounds(shape.layers * shape.hidden, 0.0f);
    std::vector<float> normalized(hidden_elements, 0.0f);

    for (std::size_t row = 0; row < rows; ++row) {
        const std::uint32_t token_id = token_ids[row];

        assert(token_id < shape.vocab);

        const float * embedding =
            weights.embedding_weight +
            static_cast<std::size_t>(token_id) * shape.hidden;

        std::copy(
            embedding,
            embedding + shape.hidden,
            current.data() + row * shape.hidden
        );
    }

    hgrn_lower_bounds_f32(
        weights.raw_lower_bounds,
        lower_bounds.data(),
        shape.layers,
        shape.hidden
    );

    for (std::size_t layer = 0; layer < shape.layers; ++layer) {
        const std::size_t conv_offset = layer * conv_state_per_layer;
        const std::size_t recurrent_offset = layer * recurrent_state_per_layer;

        hgrn_decoder_layer_f32(
            current.data(),
            weights.decoder_layers[layer],
            {
                .conv_cache = initial_state.conv_cache + conv_offset,
                .recurrent_state =
                    initial_state.recurrent_state + recurrent_offset,
            },
            next.data(),
            {
                .conv_cache = final_state.conv_cache + conv_offset,
                .recurrent_state =
                    final_state.recurrent_state + recurrent_offset,
            },
            rmsnorm_epsilon,
            {
                .batch = shape.batch,
                .sequence = shape.sequence,
                .hidden = shape.hidden,
                .heads = shape.heads,
                .intermediate = shape.intermediate,
                .conv_kernel_size = shape.conv_kernel_size,
            },
            lower_bounds.data() + layer * shape.hidden
        );

        current.swap(next);
    }

    rmsnorm_f32(
        current.data(),
        weights.final_norm_weight,
        rmsnorm_epsilon,
        normalized.data(),
        {
            .rows = rows,
            .hidden = shape.hidden,
        }
    );

    fused_bitlinear_f32(
        normalized.data(),
        weights.lm_head_norm_weight,
        weights.lm_head_weight,
        logits,
        rmsnorm_epsilon,
        {
            .rows = rows,
            .input_features = shape.hidden,
            .output_features = shape.vocab,
        }
    );
}

} // namespace terlml
