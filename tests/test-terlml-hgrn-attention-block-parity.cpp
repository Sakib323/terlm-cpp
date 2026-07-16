#include "../terlml/terlml-hgrn-attention-block.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::array<char, 8> k_magic = {
    'T', 'H', 'A', 'B', 'F', '3', '2', '\0'
};

constexpr std::uint32_t k_version = 1;
constexpr float k_tolerance = 2e-5f;

struct fixture_header {
    std::uint32_t version;
    std::uint32_t batch;
    std::uint32_t sequence;
    std::uint32_t hidden;
    std::uint32_t heads;
    std::uint32_t kernel_size;
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

bool test_hgrn_attention_block_parity(const char * path) {
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
        !read_value(stream, &header.heads, "heads") ||
        !read_value(stream, &header.kernel_size, "kernel_size") ||
        !read_value(stream, &header.epsilon, "epsilon")) {
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
        header.heads == 0 ||
        header.kernel_size == 0 ||
        header.hidden % header.heads != 0) {
        std::cerr << "Invalid fixture dimensions\n";
        return false;
    }

    const std::size_t batch = header.batch;
    const std::size_t sequence = header.sequence;
    const std::size_t hidden = header.hidden;
    const std::size_t heads = header.heads;
    const std::size_t kernel_size = header.kernel_size;
    const std::size_t head_dim = hidden / heads;

    const std::size_t input_count = batch * sequence * hidden;
    const std::size_t conv_weight_count = hidden * kernel_size;
    const std::size_t projection_count = hidden * hidden;
    const std::size_t conv_state_count = batch * hidden * kernel_size;
    const std::size_t recurrent_state_count = batch * heads * head_dim;

    std::vector<float> input;
    std::vector<float> conv_weight;
    std::vector<float> conv_bias;
    std::vector<float> i_proj_weight;
    std::vector<float> f_proj_weight;
    std::vector<float> g_proj_weight;
    std::vector<float> gnorm_weight;
    std::vector<float> o_proj_weight;
    std::vector<float> initial_conv_state;
    std::vector<float> initial_recurrent_state;
    std::vector<float> expected_output;
    std::vector<float> expected_conv_state;
    std::vector<float> expected_recurrent_state;

    if (!read_tensor(stream, &input, input_count, "input") ||
        !read_tensor(stream, &conv_weight, conv_weight_count, "conv_weight") ||
        !read_tensor(stream, &conv_bias, hidden, "conv_bias") ||
        !read_tensor(stream, &i_proj_weight, projection_count, "i_proj_weight") ||
        !read_tensor(stream, &f_proj_weight, projection_count, "f_proj_weight") ||
        !read_tensor(stream, &g_proj_weight, projection_count, "g_proj_weight") ||
        !read_tensor(stream, &gnorm_weight, hidden, "gnorm_weight") ||
        !read_tensor(stream, &o_proj_weight, projection_count, "o_proj_weight") ||
        !read_tensor(stream, &initial_conv_state, conv_state_count, "initial_conv_state") ||
        !read_tensor(stream, &initial_recurrent_state, recurrent_state_count, "initial_recurrent_state") ||
        !read_tensor(stream, &expected_output, input_count, "expected_output") ||
        !read_tensor(stream, &expected_conv_state, conv_state_count, "expected_conv_state") ||
        !read_tensor(stream, &expected_recurrent_state, recurrent_state_count, "expected_recurrent_state")) {
        return false;
    }

    std::vector<float> actual_output(input_count, 0.0f);
    std::vector<float> actual_conv_state(conv_state_count, 0.0f);
    std::vector<float> actual_recurrent_state(recurrent_state_count, 0.0f);

    const terlml::hgrn_attention_block_weights weights = {
        .conv_weight = conv_weight.data(),
        .conv_bias = conv_bias.data(),
        .i_proj_weight = i_proj_weight.data(),
        .f_proj_weight = f_proj_weight.data(),
        .g_proj_weight = g_proj_weight.data(),
        .gnorm_weight = gnorm_weight.data(),
        .o_proj_weight = o_proj_weight.data(),
    };

    terlml::hgrn_attention_block_f32(
        input.data(),
        weights,
        {
            .conv_cache = initial_conv_state.data(),
            .recurrent_state = initial_recurrent_state.data(),
        },
        actual_output.data(),
        {
            .conv_cache = actual_conv_state.data(),
            .recurrent_state = actual_recurrent_state.data(),
        },
        header.epsilon,
        {
            .batch = batch,
            .sequence = sequence,
            .hidden = hidden,
            .heads = heads,
            .conv_kernel_size = kernel_size,
        }
    );

    return expect_close("block output", actual_output, expected_output) &&
           expect_close(
               "final convolution cache",
               actual_conv_state,
               expected_conv_state
           ) &&
           expect_close(
               "final recurrent state",
               actual_recurrent_state,
               expected_recurrent_state
           );
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr
            << "Usage: test-terlml-hgrn-attention-block-parity "
            << "<fixture-path>\n";
        return EXIT_FAILURE;
    }

    if (!test_hgrn_attention_block_parity(argv[1])) {
        std::cerr << "\nHGRN attention-block parity test failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nHGRN attention-block parity test passed.\n";
    return EXIT_SUCCESS;
}
