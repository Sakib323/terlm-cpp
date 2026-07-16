#include "../terlml/terlml-rmsnorm-gate.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float k_tolerance = 1e-6f;

float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

bool expect_close(
    const std::string & name,
    const std::vector<float> & actual,
    const std::vector<float> & expected
) {
    if (actual.size() != expected.size()) {
        std::cerr << "[FAIL] " << name << ": size mismatch\n";
        return false;
    }

    float max_error = 0.0f;
    std::size_t max_index = 0;

    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float error = std::fabs(actual[index] - expected[index]);

        if (error > max_error) {
            max_error = error;
            max_index = index;
        }
    }

    if (max_error > k_tolerance) {
        std::cerr << "[FAIL] " << name
                  << ": index=" << max_index
                  << ", got=" << actual[max_index]
                  << ", expected=" << expected[max_index]
                  << ", abs_error=" << max_error << '\n';
        return false;
    }

    std::cout << "[PASS] " << name
              << ": max_abs_error=" << max_error << '\n';
    return true;
}

bool test_rmsnorm_silu_gate() {
    const terlml::rmsnorm_gate_shape shape = {
        .batch = 1,
        .sequence = 2,
        .hidden = 4,
    };

    const float epsilon = 1e-5f;

    const std::vector<float> input = {
        1.0f, -2.0f, 3.0f, -4.0f,
        2.0f, 0.0f, -1.0f, 3.0f,
    };

    const std::vector<float> gate = {
        -2.0f, -0.5f, 0.5f, 2.0f,
        1.0f, -1.0f, 0.0f, 3.0f,
    };

    const std::vector<float> weight = {
        1.0f, 0.5f, 1.5f, 2.0f,
    };

    std::vector<float> actual(input.size(), 0.0f);
    std::vector<float> expected(input.size(), 0.0f);

    for (std::size_t t = 0; t < shape.sequence; ++t) {
        const std::size_t offset = t * shape.hidden;
        float sum_of_squares = 0.0f;

        for (std::size_t d = 0; d < shape.hidden; ++d) {
            const float value = input[offset + d];
            sum_of_squares += value * value;
        }

        const float inverse_rms = 1.0f / std::sqrt(
            sum_of_squares / static_cast<float>(shape.hidden) + epsilon
        );

        for (std::size_t d = 0; d < shape.hidden; ++d) {
            expected[offset + d] =
                input[offset + d] *
                inverse_rms *
                weight[d] *
                silu(gate[offset + d]);
        }
    }

    terlml::rmsnorm_silu_gate_f32(
        gate.data(),
        input.data(),
        weight.data(),
        epsilon,
        actual.data(),
        shape
    );

    return expect_close("RMSNorm SiLU gate", actual, expected);
}

} // namespace

int main() {
    if (!test_rmsnorm_silu_gate()) {
        std::cerr << "\nRMSNorm SiLU gate tests failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nAll RMSNorm SiLU gate tests passed.\n";
    return EXIT_SUCCESS;
}
