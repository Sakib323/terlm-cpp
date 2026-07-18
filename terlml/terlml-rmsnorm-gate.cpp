#include "terlml-rmsnorm-gate.h"

#include <cmath>

namespace terlml {

namespace {

float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

} // namespace

void rmsnorm_silu_gate_f32(
    const float * gate,
    const float * input,
    const float * weight,
    float epsilon,
    float * output,
    rmsnorm_gate_shape shape
) {
    for (std::size_t b = 0; b < shape.batch; ++b) {
        for (std::size_t t = 0; t < shape.sequence; ++t) {
            const std::size_t offset =
                (b * shape.sequence + t) * shape.hidden;

            float sum_of_squares = 0.0f;

            for (std::size_t d = 0; d < shape.hidden; ++d) {
                const float value = gate[offset + d];
                sum_of_squares += value * value;
            }

            const float inverse_rms = 1.0f / std::sqrt(
                sum_of_squares / static_cast<float>(shape.hidden) + epsilon
            );

            for (std::size_t d = 0; d < shape.hidden; ++d) {
                output[offset + d] =
                    gate[offset + d] *
                    inverse_rms *
                    weight[d] *
                    silu(input[offset + d]);
            }
        }
    }
}

} // namespace terlml
