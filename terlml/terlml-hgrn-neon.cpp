#include "terlml-hgrn-neon.h"

#include <algorithm>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
#include <arm_neon.h>
#define TERLML_HAS_NEON 1
#else
#define TERLML_HAS_NEON 0
#endif

namespace terlml {

void hgrn_recurrent_f32_neon(
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
            const std::size_t state_offset =
                (b * shape.heads + h) * shape.head_dim;

            for (std::size_t t = 0; t < shape.sequence; ++t) {
                const std::size_t token_offset =
                    ((b * shape.heads + h) * shape.sequence + t) *
                    shape.head_dim;

                std::size_t d = 0;

#if TERLML_HAS_NEON
                for (; d + 4 <= shape.head_dim; d += 4) {
                    const float32x4_t x_value =
                        vld1q_f32(x + token_offset + d);
                    const float32x4_t gate_value =
                        vld1q_f32(g + token_offset + d);
                    const float32x4_t state_value =
                        vld1q_f32(state.data() + state_offset + d);

                    const float32x4_t next_state =
                        vmlaq_f32(x_value, gate_value, state_value);

                    vst1q_f32(state.data() + state_offset + d, next_state);
                    vst1q_f32(output + token_offset + d, next_state);
                }
#endif

                for (; d < shape.head_dim; ++d) {
                    const float value =
                        g[token_offset + d] * state[state_offset + d] +
                        x[token_offset + d];

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
