#include "../terlml/terlml-hgrn-mlp.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

constexpr std::array<char, 8> k_magic = {
    'T', 'H', 'M', 'L', 'P', 'F', '3', '2'
};

constexpr std::uint32_t k_version = 1;
constexpr float k_tolerance = 5e-5f;

struct fixture_header {
    std::uint32_t version;
    std::uint32_t batch;
    std::uint32_t sequence;
    std::uint32_t hidden;
    std::uint32_t intermediate;
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
    const char * name,
    const std::vector<float> & actual,
    const std::vector<float> & expected
) {
    if (actual.size() != expected.size()) {
        std::cerr << "[FAIL] " << name << ": size mismatch, got "
                  << actual.size() << ", expected " << expected.size()
                  << '\n';
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
                  << ", max_abs_error=" << max_error
                  << ", tolerance=" << k_tolerance << '\n';
        return false;
    }

    std::cout << "[PASS] " << name
              << ": max_abs_error=" << max_error << '\n';
    return true;
}

bool test_hgrn_mlp_parity(const char * path) {
    std::ifstream stream(path, std::ios::binary);

    if (!stream) {
        std::cerr << "Could not open fixture: " << path << '\n';
        return false;
    }

    std::array<char, k_magic.size()> magic{};
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));

    if (!stream || magic != k_magic) {
        std::cerr << "Invalid fixture magic\n";
        return false;
    }

    fixture_header header{};

    if (!read_value(stream, &header.version, "version") ||
        !read_value(stream, &header.batch, "batch") ||
        !read_value(stream, &header.sequence, "sequence") ||
        !read_value(stream, &header.hidden, "hidden") ||
        !read_value(stream, &header.intermediate, "intermediate")) {
        return false;
    }

    if (header.version != k_version) {
        std::cerr << "Unsupported fixture version: "
                  << header.version << '\n';
        return false;
    }

    if (header.batch == 0 ||
        header.sequence == 0 ||
        header.hidden == 0 ||
        header.intermediate == 0) {
        std::cerr << "Invalid fixture dimensions\n";
        return false;
    }

    const std::size_t batch = header.batch;
    const std::size_t sequence = header.sequence;
    const std::size_t hidden = header.hidden;
    const std::size_t intermediate = header.intermediate;
    const std::size_t rows = batch * sequence;
    const std::size_t input_count = rows * hidden;
    const std::size_t gate_weight_count = 2 * intermediate * hidden;
    const std::size_t down_weight_count = hidden * intermediate;

    std::vector<float> input;
    std::vector<float> gate_proj_norm_weight;
    std::vector<float> gate_proj_weight;
    std::vector<float> down_proj_norm_weight;
    std::vector<float> down_proj_weight;
    std::vector<float> expected_output;

    if (!read_tensor(stream, &input, input_count, "input") ||
        !read_tensor(
            stream,
            &gate_proj_norm_weight,
            hidden,
            "gate_proj_norm_weight"
        ) ||
        !read_tensor(
            stream,
            &gate_proj_weight,
            gate_weight_count,
            "gate_proj_weight"
        ) ||
        !read_tensor(
            stream,
            &down_proj_norm_weight,
            intermediate,
            "down_proj_norm_weight"
        ) ||
        !read_tensor(
            stream,
            &down_proj_weight,
            down_weight_count,
            "down_proj_weight"
        ) ||
        !read_tensor(
            stream,
            &expected_output,
            input_count,
            "expected_output"
        )) {
        return false;
    }

    std::vector<float> actual_output(input_count, 0.0f);

    terlml::hgrn_mlp_f32(
        input.data(),
        {
            .gate_proj_norm_weight = gate_proj_norm_weight.data(),
            .gate_proj_weight = gate_proj_weight.data(),
            .down_proj_norm_weight = down_proj_norm_weight.data(),
            .down_proj_weight = down_proj_weight.data(),
        },
        actual_output.data(),
        {
            .batch = batch,
            .sequence = sequence,
            .hidden = hidden,
            .intermediate = intermediate,
        }
    );

    return expect_close("MLP output", actual_output, expected_output);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr
            << "Usage: test-terlml-hgrn-mlp-parity <fixture-path>\n";
        return EXIT_FAILURE;
    }

    if (!test_hgrn_mlp_parity(argv[1])) {
        std::cerr << "\nHGRN MLP parity test failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nHGRN MLP parity test passed.\n";
    return EXIT_SUCCESS;
}
