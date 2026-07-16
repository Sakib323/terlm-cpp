#pragma once

#include <cstddef>

namespace terlml {

struct causal_conv_shape {
    std::size_t batch;
    std::size_t sequence;
    std::size_t channels;
    std::size_t kernel_size;
};

void causal_depthwise_conv1d_silu_f32(
    const float * input,
    const float * weights,
    const float * bias,
    const float * initial_cache,
    float * output,
    float * final_cache,
    causal_conv_shape shape
);

} // namespace terlml
