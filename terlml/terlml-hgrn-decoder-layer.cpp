#include "terlml-hgrn-decoder-layer.h"

#include <cassert>
#include <vector>

namespace terlml {

void hgrn_decoder_layer_f32(
    const float * input,
    const hgrn_decoder_layer_weights & weights,
    hgrn_attention_block_state initial_state,
    float * output,
    hgrn_attention_block_output_state final_state,
    float rmsnorm_epsilon,
    hgrn_decoder_layer_shape shape,
    const float * lower_bound
) {
    assert(shape.batch > 0);
    assert(shape.sequence > 0);
    assert(shape.hidden > 0);
    assert(shape.heads > 0);
    assert(shape.hidden % shape.heads == 0);
    assert(shape.intermediate > 0);
    assert(shape.conv_kernel_size > 0);

    const std::size_t elements =
        shape.batch * shape.sequence * shape.hidden;

    std::vector<float> attention_output(elements, 0.0f);
    std::vector<float> after_attention(elements, 0.0f);
    std::vector<float> mlp_output(elements, 0.0f);

    hgrn_attention_block_f32(
        input,
        weights.attention,
        initial_state,
        attention_output.data(),
        final_state,
        rmsnorm_epsilon,
        {
            .batch = shape.batch,
            .sequence = shape.sequence,
            .hidden = shape.hidden,
            .heads = shape.heads,
            .conv_kernel_size = shape.conv_kernel_size,
        },
        lower_bound
    );

    for (std::size_t index = 0; index < elements; ++index) {
        after_attention[index] = input[index] + attention_output[index];
    }

    hgrn_mlp_f32(
        after_attention.data(),
        weights.mlp,
        mlp_output.data(),
        {
            .batch = shape.batch,
            .sequence = shape.sequence,
            .hidden = shape.hidden,
            .intermediate = shape.intermediate,
        }
    );

    for (std::size_t index = 0; index < elements; ++index) {
        output[index] = after_attention[index] + mlp_output[index];
    }
}

} // namespace terlml
