#include "terlml-rmsnorm.h"

#include <cassert>
#include <cmath>

namespace terlml {

void rmsnorm_f32(
    const float * input,
    const float * weight,
    float epsilon,
    float * output,
    rmsnorm_shape shape
) {
    assert(input != nullptr);
    assert(weight != nullptr);
    assert(output != nullptr);
    assert(shape.rows > 0);
    assert(shape.hidden > 0);

    for (std::size_t row = 0; row < shape.rows; ++row) {
        const std::size_t offset = row * shape.hidden;
        float sum_of_squares = 0.0f;

        for (std::size_t channel = 0; channel < shape.hidden; ++channel) {
            const float value = input[offset + channel];
            sum_of_squares += value * value;
        }

        const float inverse_rms = 1.0f / std::sqrt(
            sum_of_squares / static_cast<float>(shape.hidden) + epsilon
        );

        for (std::size_t channel = 0; channel < shape.hidden; ++channel) {
            output[offset + channel] =
                input[offset + channel] *
                inverse_rms *
                weight[channel];
        }
    }
}

} // namespace terlml
