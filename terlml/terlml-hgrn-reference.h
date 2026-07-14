#pragma once

#include <cstddef>

namespace terlml {

struct hgrn_shape {
    std::size_t batch;
    std::size_t heads;
    std::size_t sequence;
    std::size_t head_dim;
};

void hgrn_recurrent_f32(
    const float * x,
    const float * g,
    const float * initial_state,
    float * output,
    float * final_state,
    hgrn_shape shape
);

} // namespace terlml
