#pragma once

#include "terlml-hgrn-reference.h"

namespace terlml {

void hgrn_recurrent_f32_neon(
    const float * x,
    const float * g,
    const float * initial_state,
    float * output,
    float * final_state,
    hgrn_shape shape
);

} // namespace terlml
