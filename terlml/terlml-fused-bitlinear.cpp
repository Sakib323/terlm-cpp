#include "terlml-fused-bitlinear.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace terlml {
namespace {

float round_to_nearest_even(float value) {
    return std::nearbyint(value);
}

} // namespace

void fused_bitlinear_f32(
    const float * input,
    const float * norm_weight,
    const float * linear_weight,
    float * output,
    float epsilon,
    fused_bitlinear_shape shape
) {
    float mean_absolute_weight = 0.0f;

    for (
        std::size_t index = 0;
        index < shape.output_features * shape.input_features;
        ++index
    ) {
        mean_absolute_weight += std::fabs(linear_weight[index]);
    }

    mean_absolute_weight /=
        static_cast<float>(shape.output_features * shape.input_features);

    const float ternary_scale =
        std::max(mean_absolute_weight, 1e-5f);

    std::vector<float> quantized_input(shape.input_features);

    for (std::size_t row = 0; row < shape.rows; ++row) {
        const float * input_row =
            input + row * shape.input_features;

        float sum_of_squares = 0.0f;

        for (
            std::size_t feature = 0;
            feature < shape.input_features;
            ++feature
        ) {
            sum_of_squares += input_row[feature] * input_row[feature];
        }

        const float inverse_rms = 1.0f / std::sqrt(
            sum_of_squares /
                static_cast<float>(shape.input_features) +
            epsilon
        );

        float maximum_absolute_activation = 0.0f;

        for (
            std::size_t feature = 0;
            feature < shape.input_features;
            ++feature
        ) {
            const float normalized =
                input_row[feature] *
                inverse_rms *
                norm_weight[feature];

            quantized_input[feature] = normalized;

            maximum_absolute_activation = std::max(
                maximum_absolute_activation,
                std::fabs(normalized)
            );
        }

        const float activation_scale =
            127.0f /
            std::max(maximum_absolute_activation, 1e-5f);

        for (
            std::size_t feature = 0;
            feature < shape.input_features;
            ++feature
        ) {
            const float quantized = std::clamp(
                round_to_nearest_even(
                    quantized_input[feature] * activation_scale
                ),
                -128.0f,
                127.0f
            );

            quantized_input[feature] = quantized / activation_scale;
        }

        float * output_row =
            output + row * shape.output_features;

        for (
            std::size_t output_feature = 0;
            output_feature < shape.output_features;
            ++output_feature
        ) {
            const float * weight_row =
                linear_weight +
                output_feature * shape.input_features;

            float sum = 0.0f;

            for (
                std::size_t input_feature = 0;
                input_feature < shape.input_features;
                ++input_feature
            ) {
                const float ternary_weight = std::clamp(
                    round_to_nearest_even(
                        weight_row[input_feature] / ternary_scale
                    ),
                    -1.0f,
                    1.0f
                ) * ternary_scale;

                sum +=
                    quantized_input[input_feature] *
                    ternary_weight;
            }

            output_row[output_feature] = sum;
        }
    }
}

} // namespace terlml
