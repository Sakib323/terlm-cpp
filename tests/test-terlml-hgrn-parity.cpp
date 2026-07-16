#include "../terlml/terlml-hgrn-reference.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr std::array<char, 8> k_magic = {
    'H', 'G', 'R', 'N', 'F', 'X', '0', '1'
};

constexpr std::uint32_t k_version = 1;
constexpr float k_tolerance = 1e-5f;

struct fixture_header {
    std::uint32_t version;
    std::uint32_t reserved;
    std::uint64_t batch;
    std::uint64_t heads;
    std::uint64_t sequence;
    std::uint64_t head_dim;
};

bool read_exact(
    std::ifstream & input,
    void * destination,
    std::size_t byte_count
) {
    input.read(
        static_cast<char *>(destination),
        static_cast<std::streamsize>(byte_count)
    );
    return input.good();
}

bool read_tensor(
    std::ifstream & input,
    std::vector<float> & tensor,
    std::size_t element_count
) {
    tensor.resize(element_count);
    return read_exact(input, tensor.data(), element_count * sizeof(float));
}

bool compare_tensor(
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

bool checked_product(
    std::uint64_t a,
    std::uint64_t b,
    std::size_t & result
) {
    if (a == 0 || b == 0) {
        result = 0;
        return true;
    }

    const auto maximum = static_cast<std::uint64_t>(
        std::numeric_limits<std::size_t>::max()
    );

    if (a > maximum / b) {
        return false;
    }

    result = static_cast<std::size_t>(a * b);
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    const std::filesystem::path fixture_path =
        argc > 1
            ? std::filesystem::path(argv[1])
            : std::filesystem::path("tests/fixtures/hgrn-parity-f32.bin");

    std::ifstream input(fixture_path, std::ios::binary);

    if (!input) {
        std::cerr << "[FAIL] Could not open fixture: "
                  << fixture_path << '\n';
        return EXIT_FAILURE;
    }

    std::array<char, k_magic.size()> magic = {};

    if (!read_exact(input, magic.data(), magic.size()) || magic != k_magic) {
        std::cerr << "[FAIL] Invalid fixture magic\n";
        return EXIT_FAILURE;
    }

    fixture_header header = {};

    if (!read_exact(input, &header, sizeof(header))) {
        std::cerr << "[FAIL] Could not read fixture header\n";
        return EXIT_FAILURE;
    }

    if (header.version != k_version || header.reserved != 0) {
        std::cerr << "[FAIL] Unsupported fixture version\n";
        return EXIT_FAILURE;
    }

    std::size_t state_elements = 0;
    std::size_t sequence_elements = 0;

    if (!checked_product(header.batch, header.heads, state_elements) ||
        !checked_product(state_elements, header.head_dim, state_elements) ||
        !checked_product(state_elements, header.sequence, sequence_elements)) {
        std::cerr << "[FAIL] Fixture dimensions overflow size_t\n";
        return EXIT_FAILURE;
    }

    std::vector<float> x;
    std::vector<float> g;
    std::vector<float> initial_state;
    std::vector<float> expected_output;
    std::vector<float> expected_final_state;

    if (!read_tensor(input, x, sequence_elements) ||
        !read_tensor(input, g, sequence_elements) ||
        !read_tensor(input, initial_state, state_elements) ||
        !read_tensor(input, expected_output, sequence_elements) ||
        !read_tensor(input, expected_final_state, state_elements)) {
        std::cerr << "[FAIL] Fixture is truncated\n";
        return EXIT_FAILURE;
    }

    std::vector<float> actual_output(sequence_elements, 0.0f);
    std::vector<float> actual_final_state(state_elements, 0.0f);

    const terlml::hgrn_shape shape = {
        .batch = static_cast<std::size_t>(header.batch),
        .heads = static_cast<std::size_t>(header.heads),
        .sequence = static_cast<std::size_t>(header.sequence),
        .head_dim = static_cast<std::size_t>(header.head_dim),
    };

    terlml::hgrn_recurrent_f32(
        x.data(),
        g.data(),
        initial_state.data(),
        actual_output.data(),
        actual_final_state.data(),
        shape
    );

    const bool passed =
        compare_tensor("PyTorch output parity", actual_output, expected_output) &&
        compare_tensor(
            "PyTorch final-state parity",
            actual_final_state,
            expected_final_state
        );

    if (!passed) {
        std::cerr << "\nPyTorch-to-C++ HGRN parity failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nPyTorch-to-C++ HGRN parity passed.\n";
    return EXIT_SUCCESS;
}
