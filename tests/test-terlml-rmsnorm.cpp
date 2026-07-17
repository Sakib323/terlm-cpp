#include "../terlml/terlml-rmsnorm.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr float k_tolerance = 1e-6f;

bool expect_close(
    const char * name,
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
                  << ", max_abs_error=" << max_error << '\n';
        return false;
    }

    std::cout << "[PASS] " << name
              << ": max_abs_error=" << max_error << '\n';
    return true;
}

bool test_rmsnorm() {
    const std::vector<float> input = {
         3.0f,  4.0f,
        -3.0f,  4.0f,
    };

    const std::vector<float> weight = {
        2.0f, 0.5f,
    };

    std::vector<float> actual(input.size(), 0.0f);

    terlml::rmsnorm_f32(
        input.data(),
        weight.data(),
        0.0f,
        actual.data(),
        {
            .rows = 2,
            .hidden = 2,
        }
    );

    const float inverse_rms = 1.0f / std::sqrt(12.5f);

    const std::vector<float> expected = {
         3.0f * inverse_rms * 2.0f,
         4.0f * inverse_rms * 0.5f,
        -3.0f * inverse_rms * 2.0f,
         4.0f * inverse_rms * 0.5f,
    };

    return expect_close("plain RMSNorm", actual, expected);
}

} // namespace

int main() {
    if (!test_rmsnorm()) {
        std::cerr << "\nRMSNorm tests failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nAll RMSNorm tests passed.\n";
    return EXIT_SUCCESS;
}
