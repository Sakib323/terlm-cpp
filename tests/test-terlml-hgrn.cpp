#include "../terlml/terlml-hgrn-reference.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool expect_close(
    const std::string & name,
    const std::vector<float> & actual,
    const std::vector<float> & expected,
    float tolerance = 1e-6f
) {
    if (actual.size() != expected.size()) {
        std::cerr << "[FAIL] " << name
                  << ": size mismatch, got " << actual.size()
                  << ", expected " << expected.size() << '\n';
        return false;
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float error = std::fabs(actual[i] - expected[i]);

        if (error > tolerance) {
            std::cerr << "[FAIL] " << name
                      << ": index=" << i
                      << ", got=" << actual[i]
                      << ", expected=" << expected[i]
                      << ", abs_error=" << error << '\n';
            return false;
        }
    }

    std::cout << "[PASS] " << name << '\n';
    return true;
}

bool test_zero_initial_state() {
    const terlml::hgrn_shape shape = {
        .batch = 1,
        .heads = 1,
        .sequence = 3,
        .head_dim = 1,
    };

    const std::vector<float> x = {1.0f, 2.0f, 3.0f};
    const std::vector<float> g = {0.5f, 0.5f, 0.5f};

    std::vector<float> output(3, 0.0f);
    std::vector<float> final_state(1, 0.0f);

    terlml::hgrn_recurrent_f32(
        x.data(),
        g.data(),
        nullptr,
        output.data(),
        final_state.data(),
        shape
    );

    return expect_close(
        "zero initial state output",
        output,
        {1.0f, 2.5f, 4.25f}
    ) && expect_close(
        "zero initial state final state",
        final_state,
        {4.25f}
    );
}

bool test_provided_initial_state() {
    const terlml::hgrn_shape shape = {
        .batch = 1,
        .heads = 1,
        .sequence = 3,
        .head_dim = 1,
    };

    const std::vector<float> x = {1.0f, 2.0f, 3.0f};
    const std::vector<float> g = {0.5f, 0.5f, 0.5f};
    const std::vector<float> initial_state = {8.0f};

    std::vector<float> output(3, 0.0f);
    std::vector<float> final_state(1, 0.0f);

    terlml::hgrn_recurrent_f32(
        x.data(),
        g.data(),
        initial_state.data(),
        output.data(),
        final_state.data(),
        shape
    );

    return expect_close(
        "provided initial state output",
        output,
        {5.0f, 4.5f, 5.25f}
    ) && expect_close(
        "provided initial state final state",
        final_state,
        {5.25f}
    );
}

bool test_multi_batch_head_dimension_layout() {
    const terlml::hgrn_shape shape = {
        .batch = 1,
        .heads = 2,
        .sequence = 2,
        .head_dim = 2,
    };

    const std::vector<float> x = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        10.0f, 20.0f,
        30.0f, 40.0f,
    };

    const std::vector<float> g = {
        0.5f, 0.5f,
        0.5f, 0.5f,
        0.1f, 0.2f,
        0.1f, 0.2f,
    };

    std::vector<float> output(x.size(), 0.0f);
    std::vector<float> final_state(shape.batch * shape.heads * shape.head_dim, 0.0f);

    terlml::hgrn_recurrent_f32(
        x.data(),
        g.data(),
        nullptr,
        output.data(),
        final_state.data(),
        shape
    );

    return expect_close(
        "multi-head output",
        output,
        {
            1.0f, 2.0f,
            3.5f, 5.0f,
            10.0f, 20.0f,
            31.0f, 44.0f,
        }
    ) && expect_close(
        "multi-head final state",
        final_state,
        {
            3.5f, 5.0f,
            31.0f, 44.0f,
        }
    );
}

} // namespace

int main() {
    const bool passed =
        test_zero_initial_state() &&
        test_provided_initial_state() &&
        test_multi_batch_head_dimension_layout();

    if (!passed) {
        std::cerr << "\nHGRN reference tests failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nAll HGRN reference tests passed.\n";
    return EXIT_SUCCESS;
}
