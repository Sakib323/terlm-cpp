#include "../terlml/terlml-hgrn-lower-bounds.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr float k_tolerance = 1e-6f;

bool expect_close(
    const char * name,
    float actual,
    float expected
) {
    const float error = std::fabs(actual - expected);

    if (error > k_tolerance) {
        std::cerr << "[FAIL] " << name
                  << ": got=" << actual
                  << ", expected=" << expected
                  << ", abs_error=" << error << '\n';
        return false;
    }

    return true;
}

bool test_known_values() {
    constexpr std::size_t layers = 3;
    constexpr std::size_t hidden = 2;

    const std::vector<float> raw = {
        0.0f, 0.0f,
        0.0f, std::log(2.0f),
        0.0f, std::log(3.0f),
    };

    std::vector<float> actual(layers * hidden, -1.0f);

    terlml::hgrn_lower_bounds_f32(
        raw.data(),
        actual.data(),
        layers,
        hidden
    );

    const std::vector<float> expected = {
        0.0f, 0.0f,
        1.0f / 3.0f, 1.0f / 3.0f,
        2.0f / 3.0f, 5.0f / 6.0f,
    };

    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!expect_close(
                "known lower-bound value",
                actual[index],
                expected[index]
            )) {
            return false;
        }
    }

    std::cout << "[PASS] known lower-bound values\n";
    return true;
}

bool test_invariants() {
    constexpr std::size_t layers = 4;
    constexpr std::size_t hidden = 3;

    const std::vector<float> raw = {
        -1.0f,  0.2f,  3.0f,
         0.5f, -0.4f,  1.2f,
         2.0f,  0.7f, -2.0f,
        -0.1f,  1.5f,  0.9f,
    };

    std::vector<float> output(layers * hidden, -1.0f);

    terlml::hgrn_lower_bounds_f32(
        raw.data(),
        output.data(),
        layers,
        hidden
    );

    for (std::size_t channel = 0; channel < hidden; ++channel) {
        if (!expect_close(
                "layer-zero lower bound",
                output[channel],
                0.0f
            )) {
            return false;
        }

        float previous = output[channel];

        for (std::size_t layer = 1; layer < layers; ++layer) {
            const float value = output[layer * hidden + channel];

            if (value < previous) {
                std::cerr << "[FAIL] lower bounds must be nondecreasing\n";
                return false;
            }

            if (value < 0.0f || value >= 1.0f) {
                std::cerr << "[FAIL] lower bound outside [0, 1)\n";
                return false;
            }

            previous = value;
        }
    }

    std::cout << "[PASS] lower-bound invariants\n";
    return true;
}

} // namespace

int main() {
    if (!test_known_values() || !test_invariants()) {
        std::cerr << "\nHGRN lower-bound tests failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nAll HGRN lower-bound tests passed.\n";
    return EXIT_SUCCESS;
}
