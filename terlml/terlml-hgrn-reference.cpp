#include "terlml-hgrn-reference.h"

#include <algorithm>
#include <vector>

namespace terlml {

void hgrn_recurrent_f32(
    const float * x,
    const float * g,
    const float * initial_state,
    float * output,
    float * final_state,
    hgrn_shape shape
) {
    const std::size_t state_size = shape.batch * shape.heads * shape.head_dim;
    std::vector<float> state(state_size, 0.0f);

    if (initial_state != nullptr) {
        std::copy(initial_state, initial_state + state_size, state.begin());
    }

    for (std::size_t b = 0; b < shape.batch; ++b) {
        for (std::size_t h = 0; h < shape.heads; ++h) {
            const std::size_t state_offset = (b * shape.heads + h) * shape.head_dim;

            for (std::size_t t = 0; t < shape.sequence; ++t) {
                const std::size_t token_offset =
                    ((b * shape.heads + h) * shape.sequence + t) * shape.head_dim;

                for (std::size_t d = 0; d < shape.head_dim; ++d) {
                    const float value = g[token_offset + d] * state[state_offset + d] + x[token_offset + d];

                    state[state_offset + d] = value;
                    output[token_offset + d] = value;
                }
            }
        }
    }

    if (final_state != nullptr) {
        std::copy(state.begin(), state.end(), final_state);
    }
}

} // namespace terlml
