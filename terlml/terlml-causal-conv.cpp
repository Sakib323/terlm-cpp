#include "terlml-causal-conv.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace terlml {

namespace {

float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

} // namespace

void causal_depthwise_conv1d_silu_f32(
    const float * input,
    const float * weights,
    const float * bias,
    const float * initial_cache,
    float * output,
    float * final_cache,
    causal_conv_shape shape
) {
    const std::size_t cache_elements = shape.batch * shape.channels * shape.kernel_size;
    std::vector<float> cache(cache_elements, 0.0f);

    if (initial_cache != nullptr) {
        std::copy(
            initial_cache,
            initial_cache + cache_elements,
            cache.begin()
        );
    }

    for (std::size_t b = 0; b < shape.batch; ++b) {
        for (std::size_t t = 0; t < shape.sequence; ++t) {
            for (std::size_t c = 0; c < shape.channels; ++c) {
                const std::size_t input_offset =
                    (b * shape.sequence + t) * shape.channels + c;

                const std::size_t cache_offset =
                    (b * shape.channels + c) * shape.kernel_size;

                for (std::size_t k = 0; k + 1 < shape.kernel_size; ++k) {
                    cache[cache_offset + k] =
                        cache[cache_offset + k + 1];
                }

                cache[cache_offset + shape.kernel_size - 1] =
                    input[input_offset];

                float value = 0.0f;

                for (std::size_t k = 0; k < shape.kernel_size; ++k) {
                    const std::size_t weight_offset =
                        c * shape.kernel_size + k;

                    value += cache[cache_offset + k] * weights[weight_offset];
                }

                if (bias != nullptr) {
                    value += bias[c];
                }

                output[input_offset] = silu(value);
            }
        }
    }

    if (final_cache != nullptr) {
        std::copy(cache.begin(), cache.end(), final_cache);
    }
}

} // namespace terlml
