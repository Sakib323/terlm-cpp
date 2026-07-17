#include "terlml-hgrn-attention-block.h"

#include "terlml-causal-conv.h"
#include "terlml-fused-bitlinear.h"
#include "terlml-hgrn-reference.h"
#include "terlml-rmsnorm-gate.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace terlml {

namespace {

constexpr float k_bitlinear_rmsnorm_epsilon = 1e-6f;

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

} // namespace

void hgrn_attention_block_f32(
    const float * input,
    const hgrn_attention_block_weights & weights,
    hgrn_attention_block_state initial_state,
    float * output,
    hgrn_attention_block_output_state final_state,
    float rmsnorm_epsilon,
    hgrn_attention_block_shape shape,
    const float * lower_bound
) {
    assert(shape.hidden > 0);
    assert(shape.heads > 0);
    assert(shape.hidden % shape.heads == 0);
    assert(shape.conv_kernel_size > 0);

    const std::size_t head_dim = shape.hidden / shape.heads;
    const std::size_t token_count = shape.batch * shape.sequence;
    const std::size_t hidden_elements = token_count * shape.hidden;

    const fused_bitlinear_shape projection_shape = {
        .rows = token_count,
        .input_features = shape.hidden,
        .output_features = shape.hidden,
    };

    std::vector<float> convolved(hidden_elements, 0.0f);
    std::vector<float> projected_i(hidden_elements, 0.0f);
    std::vector<float> projected_f(hidden_elements, 0.0f);
    std::vector<float> projected_g(hidden_elements, 0.0f);
    std::vector<float> recurrent_output(hidden_elements, 0.0f);
    std::vector<float> recurrent_input_bhtd(hidden_elements, 0.0f);
    std::vector<float> decay_bhtd(hidden_elements, 0.0f);
    std::vector<float> recurrent_output_bhtd(hidden_elements, 0.0f);
    std::vector<float> gated_output(hidden_elements, 0.0f);

    causal_depthwise_conv1d_silu_f32(
        input,
        weights.conv_weight,
        weights.conv_bias,
        initial_state.conv_cache,
        convolved.data(),
        final_state.conv_cache,
        {
            .batch = shape.batch,
            .sequence = shape.sequence,
            .channels = shape.hidden,
            .kernel_size = shape.conv_kernel_size,
        }
    );

    fused_bitlinear_f32(
        convolved.data(),
        weights.i_proj_norm_weight,
        weights.i_proj_weight,
        projected_i.data(),
        k_bitlinear_rmsnorm_epsilon,
        projection_shape
    );

    fused_bitlinear_f32(
        convolved.data(),
        weights.f_proj_norm_weight,
        weights.f_proj_weight,
        projected_f.data(),
        k_bitlinear_rmsnorm_epsilon,
        projection_shape
    );

    fused_bitlinear_f32(
        convolved.data(),
        weights.g_proj_norm_weight,
        weights.g_proj_weight,
        projected_g.data(),
        k_bitlinear_rmsnorm_epsilon,
        projection_shape
    );

    for (std::size_t index = 0; index < hidden_elements; ++index) {
        projected_f[index] = sigmoid(projected_f[index]);

        if (lower_bound != nullptr) {
            const float bound = lower_bound[index % shape.hidden];
            projected_f[index] =
                bound + (1.0f - bound) * projected_f[index];
        }

        projected_i[index] = silu(projected_i[index]) *
                             (1.0f - projected_f[index]);
    }

    for (std::size_t b = 0; b < shape.batch; ++b) {
        for (std::size_t h = 0; h < shape.heads; ++h) {
            for (std::size_t t = 0; t < shape.sequence; ++t) {
                for (std::size_t d = 0; d < head_dim; ++d) {
                    const std::size_t bld_index =
                        ((b * shape.sequence + t) * shape.hidden) +
                        (h * head_dim + d);

                    const std::size_t bhtd_index =
                        ((b * shape.heads + h) * shape.sequence + t) *
                        head_dim + d;

                    recurrent_input_bhtd[bhtd_index] =
                        projected_i[bld_index];

                    decay_bhtd[bhtd_index] =
                        projected_f[bld_index];
                }
            }
        }
    }

    hgrn_recurrent_f32(
        recurrent_input_bhtd.data(),
        decay_bhtd.data(),
        initial_state.recurrent_state,
        recurrent_output_bhtd.data(),
        final_state.recurrent_state,
        {
            .batch = shape.batch,
            .heads = shape.heads,
            .sequence = shape.sequence,
            .head_dim = head_dim,
        }
    );

    for (std::size_t b = 0; b < shape.batch; ++b) {
        for (std::size_t h = 0; h < shape.heads; ++h) {
            for (std::size_t t = 0; t < shape.sequence; ++t) {
                for (std::size_t d = 0; d < head_dim; ++d) {
                    const std::size_t bld_index =
                        ((b * shape.sequence + t) * shape.hidden) +
                        (h * head_dim + d);

                    const std::size_t bhtd_index =
                        ((b * shape.heads + h) * shape.sequence + t) *
                        head_dim + d;

                    recurrent_output[bld_index] =
                        recurrent_output_bhtd[bhtd_index];
                }
            }
        }
    }

    rmsnorm_silu_gate_f32(
        projected_g.data(),
        recurrent_output.data(),
        weights.gnorm_weight,
        rmsnorm_epsilon,
        gated_output.data(),
        {
            .batch = shape.batch,
            .sequence = shape.sequence,
            .hidden = shape.hidden,
        }
    );

    fused_bitlinear_f32(
        gated_output.data(),
        weights.o_proj_norm_weight,
        weights.o_proj_weight,
        output,
        k_bitlinear_rmsnorm_epsilon,
        projection_shape
    );
}

} // namespace terlml
