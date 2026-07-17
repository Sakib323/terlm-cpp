#include "../terlml/terlml-fused-bitlinear.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::array<char, 8> k_magic = {
    'T', 'F', 'B', 'L', 'F', '3', '2', '\0'
};

constexpr std::uint32_t k_version = 1;
constexpr float k_tolerance = 2e-5f;

struct fixture_header {
    std::uint32_t version;
    std::uint32_t rows;
    std::uint32_t input_features;
    std::uint32_t output_features;
    float epsilon;
};

template <typename T>
bool read_value(
    std::ifstream & stream,
    T * value,
    const char * name
) {
    stream.read(
        reinterpret_cast<char *>(value),
        static_cast<std::streamsize>(sizeof(T))
    );

    if (!stream) {
        std::cerr << "Failed to read " << name << '\n';
        return false;
    }

    return true;
}

bool read_tensor(
    std::ifstream & stream,
    std::vector<float> * values,
    std::size_t count,
    const char * name
) {
    values->resize(count);

    stream.read(
        reinterpret_cast<char *>(values->data()),
        static_cast<std::streamsize>(count * sizeof(float))
    );

    if (!stream) {
        std::cerr << "Failed to read tensor " << name << '\n';
        return false;
    }

    return true;
}

bool expect_close(
    const std::vector<float> & actual,
    const std::vector<float> & expected
) {
    if (actual.size() != expected.size()) {
        std::cerr << "Output size mismatch\n";
        return false;
    }

    float maximum_error = 0.0f;
    std::size_t maximum_error_index = 0;

    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float error = std::fabs(actual[index] - expected[index]);

        if (error > maximum_error) {
            maximum_error = error;
            maximum_error_index = index;
        }
    }

    std::cout
        << "max_abs_error=" << maximum_error
        << " at index=" << maximum_error_index
        << '\n';

    if (maximum_error > k_tolerance) {
        std::cerr
            << "Mismatch: got=" << actual[maximum_error_index]
            << ", expected=" << expected[maximum_error_index]
            << ", tolerance=" << k_tolerance
            << '\n';
        return false;
    }

    return true;
}

bool run_test(const char * fixture_path) {
    std::ifstream stream(fixture_path, std::ios::binary);

    if (!stream) {
        std::cerr << "Could not open fixture: "
                  << fixture_path << '\n';
        return false;
    }

    std::array<char, k_magic.size()> magic{};
    stream.read(
        magic.data(),
        static_cast<std::streamsize>(magic.size())
    );

    if (!stream || magic != k_magic) {
        std::cerr << "Invalid fixture magic\n";
        return false;
    }

    fixture_header header{};

    if (!read_value(stream, &header.version, "version") ||
        !read_value(stream, &header.rows, "rows") ||
        !read_value(stream, &header.input_features, "input_features") ||
        !read_value(stream, &header.output_features, "output_features") ||
        !read_value(stream, &header.epsilon, "epsilon")) {
        return false;
    }

    if (header.version != k_version ||
        header.rows == 0 ||
        header.input_features == 0 ||
        header.output_features == 0) {
        std::cerr << "Invalid fixture header\n";
        return false;
    }

    const std::size_t input_count =
        static_cast<std::size_t>(header.rows) *
        header.input_features;

    const std::size_t weight_count =
        static_cast<std::size_t>(header.output_features) *
        header.input_features;

    const std::size_t output_count =
        static_cast<std::size_t>(header.rows) *
        header.output_features;

    std::vector<float> input;
    std::vector<float> norm_weight;
    std::vector<float> linear_weight;
    std::vector<float> expected_output;

    if (!read_tensor(stream, &input, input_count, "input") ||
        !read_tensor(
            stream,
            &norm_weight,
            header.input_features,
            "norm_weight"
        ) ||
        !read_tensor(
            stream,
            &linear_weight,
            weight_count,
            "linear_weight"
        ) ||
        !read_tensor(
            stream,
            &expected_output,
            output_count,
            "expected_output"
        )) {
        return false;
    }

    std::vector<float> actual_output(output_count, 0.0f);

    terlml::fused_bitlinear_f32(
        input.data(),
        norm_weight.data(),
        linear_weight.data(),
        actual_output.data(),
        header.epsilon,
        {
            .rows = header.rows,
            .input_features = header.input_features,
            .output_features = header.output_features,
        }
    );

    return expect_close(actual_output, expected_output);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr
            << "Usage: test-terlml-fused-bitlinear-parity "
            << "<fixture-path>\n";
        return EXIT_FAILURE;
    }

    return run_test(argv[1]) ? EXIT_SUCCESS : EXIT_FAILURE;
}
