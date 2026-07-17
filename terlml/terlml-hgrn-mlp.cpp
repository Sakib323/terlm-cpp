#include "terlml-hgrn-mlp.h"

#include "terlml-fused-bitlinear.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace terlml {

namespace {

constexpr float k_bitlinear_rmsnorm_epsilon = 1e-6f;

float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

} // namespace

void hgrn_mlp_f32(
    const float * input,
    const hgrn_mlp_weights & weights,
    float * output,
    hgrn_mlp_shape shape
) {
    assert(shape.hidden > 0);
    assert(shape.intermediate > 0);

    const std::size_t rows = shape.batch * shape.sequence;
    const std::size_t gate_features = 2 * shape.intermediate;

    std::vector<float> gate_and_value(rows * gate_features, 0.0f);
    std::vector<float> swiglu_output(rows * shape.intermediate, 0.0f);

    fused_bitlinear_f32(
        input,
        weights.gate_proj_norm_weight,
        weights.gate_proj_weight,
        gate_and_value.data(),
        k_bitlinear_rmsnorm_epsilon,
        {
            .rows = rows,
            .input_features = shape.hidden,
            .output_features = gate_features,
        }
    );

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t gate_base = row * gate_features;
        const std::size_t output_base = row * shape.intermediate;

        for (std::size_t channel = 0;
             channel < shape.intermediate;
             ++channel) {
            swiglu_output[output_base + channel] =
                silu(gate_and_value[gate_base + channel]) *
                gate_and_value[
                    gate_base + shape.intermediate + channel
                ];
        }
    }

    fused_bitlinear_f32(
        swiglu_output.data(),
        weights.down_proj_norm_weight,
        weights.down_proj_weight,
        output,
        k_bitlinear_rmsnorm_epsilon,
        {
            .rows = rows,
            .input_features = shape.intermediate,
            .output_features = shape.hidden,
        }
    );
}

} // namespace terlml
