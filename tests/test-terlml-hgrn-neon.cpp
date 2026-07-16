#include "../terlml/terlml-hgrn-neon.h"
#include "../terlml/terlml-hgrn-reference.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float k_tolerance = 1e-5f;

float deterministic_value(std::size_t index, float scale) {
    const std::uint32_t value =
        static_cast<std::uint32_t>((index * 1664525ULL + 1013904223ULL) &
                                   0xffffffffULL);

    const float normalized =
        static_cast<float>(value % 10000U) / 10000.0f;

    return (normalized * 2.0f - 1.0f) * scale;
}

bool compare(
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
                  << ": max_abs_error=" << max_error
                  << ", index=" << max_index
                  << ", got=" << actual[max_index]
                  << ", expected=" << expected[max_index]
                  << '\n';
        return false;
    }

    std::cout << "[PASS] " << name
              << ": max_abs_error=" << max_error << '\n';
    return true;
}

bool run_case(bool use_initial_state) {
    const terlml::hgrn_shape shape = {
        .batch = 2,
        .heads = 3,
        .sequence = 7,
        .head_dim = 13,
    };

    const std::size_t state_elements =
        shape.batch * shape.heads * shape.head_dim;

    const std::size_t sequence_elements =
        state_elements * shape.sequence;

    std::vector<float> x(sequence_elements);
    std::vector<float> g(sequence_elements);
    std::vector<float> initial_state(state_elements);

    for (std::size_t index = 0; index < sequence_elements; ++index) {
        x[index] = deterministic_value(index, 2.0f);
        g[index] = 0.5f + deterministic_value(index + 41, 0.45f);
    }

    for (std::size_t index = 0; index < state_elements; ++index) {
        initial_state[index] = deterministic_value(index + 97, 1.0f);
    }

    std::vector<float> reference_output(sequence_elements, 0.0f);
    std::vector<float> reference_final_state(state_elements, 0.0f);
    std::vector<float> neon_output(sequence_elements, 0.0f);
    std::vector<float> neon_final_state(state_elements, 0.0f);

    const float * state_pointer =
        use_initial_state ? initial_state.data() : nullptr;

    terlml::hgrn_recurrent_f32(
        x.data(),
        g.data(),
        state_pointer,
        reference_output.data(),
        reference_final_state.data(),
        shape
    );

    terlml::hgrn_recurrent_f32_neon(
        x.data(),
        g.data(),
        state_pointer,
        neon_output.data(),
        neon_final_state.data(),
        shape
    );

    const std::string prefix =
        use_initial_state ? "NEON with initial state" : "NEON zero state";

    return compare(
        prefix + " output",
        neon_output,
        reference_output
    ) && compare(
        prefix + " final state",
        neon_final_state,
        reference_final_state
    );
}

} // namespace

int main() {
    const bool passed =
        run_case(false) &&
        run_case(true);

    if (!passed) {
        std::cerr << "\nHGRN NEON parity tests failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nAll HGRN NEON parity tests passed.\n";
    return EXIT_SUCCESS;
}
