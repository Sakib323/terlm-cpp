#include "../terlml/terlml-hgrn-model.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

constexpr std::array<char, 8> k_magic = {
    'T', 'H', 'M', 'D', 'F', '3', '2', '\0'
};

constexpr std::uint32_t k_version = 1;
constexpr float k_tolerance = 2e-4f;

struct fixture_header {
    std::uint32_t version;
    std::uint32_t batch;
    std::uint32_t sequence;
    std::uint32_t vocab;
    std::uint32_t layers;
    std::uint32_t hidden;
    std::uint32_t heads;
    std::uint32_t intermediate;
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

template <typename T>
bool read_tensor(
    std::ifstream & stream,
    std::vector<T> * values,
    std::size_t count,
    const char * name
) {
    values->resize(count);

    stream.read(
        reinterpret_cast<char *>(values->data()),
        static_cast<std::streamsize>(count * sizeof(T))
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
        std::cerr << "[FAIL] " << name
                  << ": size mismatch, got " << actual.size()
                  << ", expected " << expected.size() << '\n';
        return false;
    }

    float maximum_error = 0.0f;
    std::size_t maximum_index = 0;

    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float error = std::fabs(actual[index] - expected[index]);

        if (error > maximum_error) {
            maximum_error = error;
            maximum_index = index;
        }
    }

    if (maximum_error > k_tolerance) {
        std::cerr << "[FAIL] " << name
                  << ": index=" << maximum_index
                  << ", got=" << actual[maximum_index]
                  << ", expected=" << expected[maximum_index]
                  << ", max_abs_error=" << maximum_error
                  << ", tolerance=" << k_tolerance << '\n';
        return false;
    }

    std::cout << "[PASS] " << name
              << ": max_abs_error=" << maximum_error << '\n';
    return true;
}

bool test_hgrn_model_parity(const char * path) {
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
        !read_value(stream, &header.vocab, "vocab") ||
        !read_value(stream, &header.layers, "layers") ||
        !read_value(stream, &header.hidden, "hidden") ||
        !read_value(stream, &header.heads, "heads") ||
        !read_value(stream, &header.intermediate, "intermediate") ||
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
        header.vocab == 0 ||
        header.layers == 0 ||
        header.hidden == 0 ||
        header.heads == 0 ||
        header.intermediate == 0 ||
        header.kernel_size == 0 ||
        header.hidden % header.heads != 0) {
        std::cerr << "Invalid fixture dimensions\n";
        return false;
    }

    const std::size_t batch = header.batch;
    const std::size_t sequence = header.sequence;
    const std::size_t vocab = header.vocab;
    const std::size_t layers = header.layers;
    const std::size_t hidden = header.hidden;
    const std::size_t heads = header.heads;
    const std::size_t intermediate = header.intermediate;
    const std::size_t kernel_size = header.kernel_size;
    const std::size_t rows = batch * sequence;

    const std::size_t token_count = rows;
    const std::size_t embedding_count = vocab * hidden;
    const std::size_t lower_bounds_count = layers * hidden;
    const std::size_t projection_count = hidden * hidden;
    const std::size_t conv_weight_count = hidden * kernel_size;
    const std::size_t gate_projection_count = 2 * intermediate * hidden;
    const std::size_t down_projection_count = hidden * intermediate;
    const std::size_t conv_state_count =
        layers * batch * hidden * kernel_size;
    const std::size_t recurrent_state_count = layers * batch * hidden;
    const std::size_t logits_count = rows * vocab;

    std::vector<std::uint32_t> token_ids;
    std::vector<float> embedding_weight;
    std::vector<float> raw_lower_bounds;
    std::vector<float> final_norm_weight;
    std::vector<float> lm_head_norm_weight;
    std::vector<float> lm_head_weight;

    std::vector<float> conv_weights(layers * conv_weight_count);
    std::vector<float> conv_biases(layers * hidden);
    std::vector<float> i_proj_norm_weights(layers * hidden);
    std::vector<float> i_proj_weights(layers * projection_count);
    std::vector<float> f_proj_norm_weights(layers * hidden);
    std::vector<float> f_proj_weights(layers * projection_count);
    std::vector<float> g_proj_norm_weights(layers * hidden);
    std::vector<float> g_proj_weights(layers * projection_count);
    std::vector<float> gnorm_weights(layers * hidden);
    std::vector<float> o_proj_norm_weights(layers * hidden);
    std::vector<float> o_proj_weights(layers * projection_count);
    std::vector<float> gate_proj_norm_weights(layers * hidden);
    std::vector<float> gate_proj_weights(layers * gate_projection_count);
    std::vector<float> down_proj_norm_weights(layers * intermediate);
    std::vector<float> down_proj_weights(layers * down_projection_count);

    std::vector<float> initial_conv_state;
    std::vector<float> initial_recurrent_state;
    std::vector<float> expected_logits;
    std::vector<float> expected_conv_state;
    std::vector<float> expected_recurrent_state;

    if (!read_tensor(stream, &token_ids, token_count, "token_ids") ||
        !read_tensor(
            stream, &embedding_weight, embedding_count, "embedding_weight"
        ) ||
        !read_tensor(
            stream, &raw_lower_bounds, lower_bounds_count, "raw_lower_bounds"
        ) ||
        !read_tensor(
            stream, &final_norm_weight, hidden, "final_norm_weight"
        ) ||
        !read_tensor(
            stream, &lm_head_norm_weight, hidden, "lm_head_norm_weight"
        ) ||
        !read_tensor(
            stream, &lm_head_weight, vocab * hidden, "lm_head_weight"
        )) {
        return false;
    }

    auto read_layer_tensor = [&stream](
        std::vector<float> * destination,
        std::size_t offset,
        std::size_t count,
        const char * name
    ) {
        std::vector<float> values;

        if (!read_tensor(stream, &values, count, name)) {
            return false;
        }

        std::copy(
            values.begin(),
            values.end(),
            destination->begin() + static_cast<std::ptrdiff_t>(offset)
        );

        return true;
    };

    for (std::size_t layer = 0; layer < layers; ++layer) {
        if (!read_layer_tensor(
                &conv_weights,
                layer * conv_weight_count,
                conv_weight_count,
                "conv_weight"
            ) ||
            !read_layer_tensor(
                &conv_biases, layer * hidden, hidden, "conv_bias"
            ) ||
            !read_layer_tensor(
                &i_proj_norm_weights,
                layer * hidden,
                hidden,
                "i_proj_norm_weight"
            ) ||
            !read_layer_tensor(
                &i_proj_weights,
                layer * projection_count,
                projection_count,
                "i_proj_weight"
            ) ||
            !read_layer_tensor(
                &f_proj_norm_weights,
                layer * hidden,
                hidden,
                "f_proj_norm_weight"
            ) ||
            !read_layer_tensor(
                &f_proj_weights,
                layer * projection_count,
                projection_count,
                "f_proj_weight"
            ) ||
            !read_layer_tensor(
                &g_proj_norm_weights,
                layer * hidden,
                hidden,
                "g_proj_norm_weight"
            ) ||
            !read_layer_tensor(
                &g_proj_weights,
                layer * projection_count,
                projection_count,
                "g_proj_weight"
            ) ||
            !read_layer_tensor(
                &gnorm_weights, layer * hidden, hidden, "gnorm_weight"
            ) ||
            !read_layer_tensor(
                &o_proj_norm_weights,
                layer * hidden,
                hidden,
                "o_proj_norm_weight"
            ) ||
            !read_layer_tensor(
                &o_proj_weights,
                layer * projection_count,
                projection_count,
                "o_proj_weight"
            ) ||
            !read_layer_tensor(
                &gate_proj_norm_weights,
                layer * hidden,
                hidden,
                "gate_proj_norm_weight"
            ) ||
            !read_layer_tensor(
                &gate_proj_weights,
                layer * gate_projection_count,
                gate_projection_count,
                "gate_proj_weight"
            ) ||
            !read_layer_tensor(
                &down_proj_norm_weights,
                layer * intermediate,
                intermediate,
                "down_proj_norm_weight"
            ) ||
            !read_layer_tensor(
                &down_proj_weights,
                layer * down_projection_count,
                down_projection_count,
                "down_proj_weight"
            )) {
            return false;
        }
    }

    if (!read_tensor(
            stream, &initial_conv_state, conv_state_count, "initial_conv_state"
        ) ||
        !read_tensor(
            stream,
            &initial_recurrent_state,
            recurrent_state_count,
            "initial_recurrent_state"
        ) ||
        !read_tensor(stream, &expected_logits, logits_count, "expected_logits") ||
        !read_tensor(
            stream, &expected_conv_state, conv_state_count, "expected_conv_state"
        ) ||
        !read_tensor(
            stream,
            &expected_recurrent_state,
            recurrent_state_count,
            "expected_recurrent_state"
        )) {
        return false;
    }

    std::vector<terlml::hgrn_decoder_layer_weights> decoder_layers(layers);

    for (std::size_t layer = 0; layer < layers; ++layer) {
        decoder_layers[layer] = {
            .attention = {
                .conv_weight =
                    conv_weights.data() + layer * conv_weight_count,
                .conv_bias = conv_biases.data() + layer * hidden,
                .i_proj_norm_weight =
                    i_proj_norm_weights.data() + layer * hidden,
                .i_proj_weight =
                    i_proj_weights.data() + layer * projection_count,
                .f_proj_norm_weight =
                    f_proj_norm_weights.data() + layer * hidden,
                .f_proj_weight =
                    f_proj_weights.data() + layer * projection_count,
                .g_proj_norm_weight =
                    g_proj_norm_weights.data() + layer * hidden,
                .g_proj_weight =
                    g_proj_weights.data() + layer * projection_count,
                .gnorm_weight = gnorm_weights.data() + layer * hidden,
                .o_proj_norm_weight =
                    o_proj_norm_weights.data() + layer * hidden,
                .o_proj_weight =
                    o_proj_weights.data() + layer * projection_count,
            },
            .mlp = {
                .gate_proj_norm_weight =
                    gate_proj_norm_weights.data() + layer * hidden,
                .gate_proj_weight =
                    gate_proj_weights.data() + layer * gate_projection_count,
                .down_proj_norm_weight =
                    down_proj_norm_weights.data() + layer * intermediate,
                .down_proj_weight =
                    down_proj_weights.data() + layer * down_projection_count,
            },
        };
    }

    std::vector<float> actual_logits(logits_count, 0.0f);
    std::vector<float> actual_conv_state(conv_state_count, 0.0f);
    std::vector<float> actual_recurrent_state(recurrent_state_count, 0.0f);

    terlml::hgrn_model_f32(
        token_ids.data(),
        {
            .embedding_weight = embedding_weight.data(),
            .decoder_layers = decoder_layers.data(),
            .raw_lower_bounds = raw_lower_bounds.data(),
            .final_norm_weight = final_norm_weight.data(),
            .lm_head_norm_weight = lm_head_norm_weight.data(),
            .lm_head_weight = lm_head_weight.data(),
        },
        {
            .conv_cache = initial_conv_state.data(),
            .recurrent_state = initial_recurrent_state.data(),
        },
        actual_logits.data(),
        {
            .conv_cache = actual_conv_state.data(),
            .recurrent_state = actual_recurrent_state.data(),
        },
        header.epsilon,
        {
            .batch = batch,
            .sequence = sequence,
            .vocab = vocab,
            .layers = layers,
            .hidden = hidden,
            .heads = heads,
            .intermediate = intermediate,
            .conv_kernel_size = kernel_size,
        }
    );

    return expect_close("model logits", actual_logits, expected_logits) &&
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
            << "Usage: test-terlml-hgrn-model-parity <fixture-path>\n";
        return EXIT_FAILURE;
    }

    if (!test_hgrn_model_parity(argv[1])) {
        std::cerr << "\nHGRN model parity test failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nHGRN model parity test passed.\n";
    return EXIT_SUCCESS;
}
