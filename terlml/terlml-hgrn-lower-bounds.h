#pragma once

#include <cstddef>

namespace terlml {

void hgrn_lower_bounds_f32(
    const float * raw_lower_bounds,
    float * output_lower_bounds,
    std::size_t layers,
    std::size_t hidden
);

} // namespace terlml
