#include "../terlml/terlml-causal-conv.h"

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

    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float error = std::fabs(actual[index] - expected[index]);

        if (error > k_tolerance) {
            std::cerr << "[FAIL] " << name
                      << ": index=" << index
                      << ", got=" << actual[index]
                      << ", expected=" << expected[index]
                      << ", abs_error=" << error << '\n';
            return false;
        }
    }

    std::cout << "[PASS] " << name << '\n';
    return true;
}

bool test_causal_cache_continuation() {
    const terlml::causal_conv_shape shape = {
        .batch = 1,
        .sequence = 3,
        .channels = 1,
        .kernel_size = 3,
    };

    const std::vector<float> weights = {0.25f, 0.5f, 1.0f};
    const std::vector<float> bias = {0.1f};
    const std::vector<float> input = {1.0f, 2.0f, 3.0f};

    std::vector<float> output(3, 0.0f);
    std::vector<float> final_cache(3, 0.0f);

    terlml::causal_depthwise_conv1d_silu_f32(
        input.data(),
        weights.data(),
        bias.data(),
        nullptr,
        output.data(),
        final_cache.data(),
        shape
    );

    const std::vector<float> expected_output = {
        silu(1.1f),
        silu(2.6f),
        silu(4.35f),
    };

    const std::vector<float> expected_cache = {
        1.0f, 2.0f, 3.0f,
    };

    const terlml::causal_conv_shape decode_shape = {
        .batch = 1,
        .sequence = 1,
        .channels = 1,
        .kernel_size = 3,
    };

    const std::vector<float> decode_input = {4.0f};
    std::vector<float> decode_output(1, 0.0f);
    std::vector<float> decode_final_cache(3, 0.0f);

    terlml::causal_depthwise_conv1d_silu_f32(
        decode_input.data(),
        weights.data(),
        bias.data(),
        final_cache.data(),
        decode_output.data(),
        decode_final_cache.data(),
        decode_shape
    );

    return expect_close("prefill output", output, expected_output) &&
           expect_close("prefill cache", final_cache, expected_cache) &&
           expect_close(
               "decode output",
               decode_output,
               {silu(6.1f)}
           ) &&
           expect_close(
               "decode cache",
               decode_final_cache,
               {2.0f, 3.0f, 4.0f}
           );
}

} // namespace

int main() {
    if (!test_causal_cache_continuation()) {
        std::cerr << "\nCausal convolution tests failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nAll causal convolution tests passed.\n";
    return EXIT_SUCCESS;
}
