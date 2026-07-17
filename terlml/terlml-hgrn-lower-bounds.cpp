#include "terlml-hgrn-lower-bounds.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

namespace terlml {

void hgrn_lower_bounds_f32(
    const float * raw_lower_bounds,
    float * output_lower_bounds,
    std::size_t layers,
    std::size_t hidden
) {
    assert(raw_lower_bounds != nullptr);
    assert(output_lower_bounds != nullptr);
    assert(layers > 0);
    assert(hidden > 0);

    std::vector<float> probabilities(layers);

    for (std::size_t channel = 0; channel < hidden; ++channel) {
        float maximum = -std::numeric_limits<float>::infinity();

        for (std::size_t layer = 0; layer < layers; ++layer) {
            const float value = raw_lower_bounds[layer * hidden + channel];
            maximum = value > maximum ? value : maximum;
        }

        float denominator = 0.0f;

        for (std::size_t layer = 0; layer < layers; ++layer) {
            const float value = raw_lower_bounds[layer * hidden + channel];
            const float probability = std::exp(value - maximum);

            probabilities[layer] = probability;
            denominator += probability;
        }

        float cumulative = 0.0f;
        float first_cumulative = 0.0f;

        for (std::size_t layer = 0; layer < layers; ++layer) {
            cumulative += probabilities[layer] / denominator;

            if (layer == 0) {
                first_cumulative = cumulative;
            }

            output_lower_bounds[layer * hidden + channel] =
                cumulative - first_cumulative;
        }
    }
}

} // namespace terlml
