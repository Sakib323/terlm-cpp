#include "../terlml/terlml-fused-bitlinear.h"
#include "../terlml/terlml-hgrn-decoder-layer.h"
#include "../terlml/terlml-hgrn-lower-bounds.h"
#include "../terlml/terlml-hgrn-model.h"
#include "../terlml/terlml-rmsnorm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float k_tolerance = 1e-5f;

float deterministic_value(std::size_t index, float scale) {
    const std::uint32_t value = static_cast<std::uint32_t>(
        (index * 1664525ULL + 1013904223ULL) & 0xffffffffULL
    );

    return (
        static_cast<float>(value % 10000U) / 5000.0f - 1.0f
    ) * scale;
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
                  << ", max_abs_error=" << maximum_error << '\n';
        return false;
    }

    std::cout << "[PASS] " << name
              << ": max_abs_error=" << maximum_error << '\n';
    return true;
}

std::vector<std::uint32_t> slice_ids(
    const std::vector<std::uint32_t> & input,
    std::size_t start,
    std::size_t count
) {
    return {
        input.begin() + static_cast<std::ptrdiff_t>(start),
        input.begin() + static_cast<std::ptrdiff_t>(start + count),
    };
}

std::vector<float> join_rows(
    const std::vector<float> & first,
    const std::vector<float> & second
) {
    std::vector<float> joined;
    joined.reserve(first.size() + second.size());
    joined.insert(joined.end(), first.begin(), first.end());
    joined.insert(joined.end(), second.begin(), second.end());
    return joined;
}

bool test_model_composition_and_continuation() {
    constexpr std::size_t batch = 1;
    constexpr std::size_t sequence = 5;
    constexpr std::size_t first_sequence = 2;
    constexpr std::size_t second_sequence = sequence - first_sequence;
    constexpr std::size_t vocab = 17;
    constexpr std::size_t layers = 2;
    constexpr std::size_t hidden = 12;
    constexpr std::size_t heads = 3;
    constexpr std::size_t intermediate = 20;
    constexpr std::size_t kernel_size = 3;
    constexpr float epsilon = 1e-5f;

    const std::size_t rows = batch * sequence;
    const std::size_t hidden_elements = rows * hidden;
    const std::size_t conv_state_per_layer = batch * hidden * kernel_size;
    const std::size_t recurrent_state_per_layer = batch * hidden;
    const std::size_t total_conv_state = layers * conv_state_per_layer;
    const std::size_t total_recurrent_state = layers * recurrent_state_per_layer;
    const std::size_t projection_elements = hidden * hidden;
    const std::size_t gate_weight_elements = 2 * intermediate * hidden;
    const std::size_t down_weight_elements = hidden * intermediate;

    std::vector<std::uint32_t> token_ids = {3, 1, 14, 5, 9};
    std::vector<float> embedding_weight(vocab * hidden);
    std::vector<float> raw_lower_bounds(layers * hidden);
    std::vector<float> final_norm_weight(hidden);
    std::vector<float> lm_head_norm_weight(hidden);
    std::vector<float> lm_head_weight(vocab * hidden);

    std::vector<float> conv_weights(layers * hidden * kernel_size);
    std::vector<float> conv_biases(layers * hidden);
    std::vector<float> i_norm_weights(layers * hidden);
    std::vector<float> i_weights(layers * projection_elements);
    std::vector<float> f_norm_weights(layers * hidden);
    std::vector<float> f_weights(layers * projection_elements);
    std::vector<float> g_norm_weights(layers * hidden);
    std::vector<float> g_weights(layers * projection_elements);
    std::vector<float> gnorm_weights(layers * hidden);
    std::vector<float> o_norm_weights(layers * hidden);
    std::vector<float> o_weights(layers * projection_elements);
    std::vector<float> mlp_gate_norm_weights(layers * hidden);
    std::vector<float> mlp_gate_weights(layers * gate_weight_elements);
    std::vector<float> mlp_down_norm_weights(layers * intermediate);
    std::vector<float> mlp_down_weights(layers * down_weight_elements);

    std::vector<float> initial_conv_state(total_conv_state);
    std::vector<float> initial_recurrent_state(total_recurrent_state);

    auto fill = [](std::vector<float> & values, std::size_t offset, float scale) {
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = deterministic_value(index + offset, scale);
        }
    };

    fill(embedding_weight, 101, 0.7f);
    fill(raw_lower_bounds, 201, 0.4f);
    fill(final_norm_weight, 301, 0.15f);
    fill(lm_head_norm_weight, 401, 0.15f);
    fill(lm_head_weight, 501, 0.15f);
    fill(conv_weights, 601, 0.2f);
    fill(conv_biases, 701, 0.1f);
    fill(i_norm_weights, 801, 0.15f);
    fill(i_weights, 901, 0.15f);
    fill(f_norm_weights, 1001, 0.15f);
    fill(f_weights, 1101, 0.15f);
    fill(g_norm_weights, 1201, 0.15f);
    fill(g_weights, 1301, 0.15f);
    fill(gnorm_weights, 1401, 0.15f);
    fill(o_norm_weights, 1501, 0.15f);
    fill(o_weights, 1601, 0.15f);
    fill(mlp_gate_norm_weights, 1701, 0.15f);
    fill(mlp_gate_weights, 1801, 0.15f);
    fill(mlp_down_norm_weights, 1901, 0.15f);
    fill(mlp_down_weights, 2001, 0.15f);
    fill(initial_conv_state, 2101, 0.4f);
    fill(initial_recurrent_state, 2201, 0.4f);

    for (std::size_t index = 0; index < hidden; ++index) {
        final_norm_weight[index] += 0.9f;
        lm_head_norm_weight[index] += 0.9f;
    }

    for (std::size_t layer = 0; layer < layers; ++layer) {
        for (std::size_t index = 0; index < hidden; ++index) {
            i_norm_weights[layer * hidden + index] += 0.9f;
            f_norm_weights[layer * hidden + index] += 0.9f;
            g_norm_weights[layer * hidden + index] += 0.9f;
            gnorm_weights[layer * hidden + index] += 0.9f;
            o_norm_weights[layer * hidden + index] += 0.9f;
            mlp_gate_norm_weights[layer * hidden + index] += 0.9f;
        }

        for (std::size_t index = 0; index < intermediate; ++index) {
            mlp_down_norm_weights[layer * intermediate + index] += 0.9f;
        }
    }

    std::vector<terlml::hgrn_decoder_layer_weights> decoder_layers(layers);

    for (std::size_t layer = 0; layer < layers; ++layer) {
        decoder_layers[layer] = {
            .attention = {
                .conv_weight =
                    conv_weights.data() + layer * hidden * kernel_size,
                .conv_bias = conv_biases.data() + layer * hidden,
                .i_proj_norm_weight =
                    i_norm_weights.data() + layer * hidden,
                .i_proj_weight =
                    i_weights.data() + layer * projection_elements,
                .f_proj_norm_weight =
                    f_norm_weights.data() + layer * hidden,
                .f_proj_weight =
                    f_weights.data() + layer * projection_elements,
                .g_proj_norm_weight =
                    g_norm_weights.data() + layer * hidden,
                .g_proj_weight =
                    g_weights.data() + layer * projection_elements,
                .gnorm_weight =
                    gnorm_weights.data() + layer * hidden,
                .o_proj_norm_weight =
                    o_norm_weights.data() + layer * hidden,
                .o_proj_weight =
                    o_weights.data() + layer * projection_elements,
            },
            .mlp = {
                .gate_proj_norm_weight =
                    mlp_gate_norm_weights.data() + layer * hidden,
                .gate_proj_weight =
                    mlp_gate_weights.data() + layer * gate_weight_elements,
                .down_proj_norm_weight =
                    mlp_down_norm_weights.data() + layer * intermediate,
                .down_proj_weight =
                    mlp_down_weights.data() + layer * down_weight_elements,
            },
        };
    }

    const terlml::hgrn_model_weights weights = {
        .embedding_weight = embedding_weight.data(),
        .decoder_layers = decoder_layers.data(),
        .raw_lower_bounds = raw_lower_bounds.data(),
        .final_norm_weight = final_norm_weight.data(),
        .lm_head_norm_weight = lm_head_norm_weight.data(),
        .lm_head_weight = lm_head_weight.data(),
    };

    const terlml::hgrn_model_shape shape = {
        .batch = batch,
        .sequence = sequence,
        .vocab = vocab,
        .layers = layers,
        .hidden = hidden,
        .heads = heads,
        .intermediate = intermediate,
        .conv_kernel_size = kernel_size,
    };

    std::vector<float> full_logits(rows * vocab, 0.0f);
    std::vector<float> full_conv_state(total_conv_state, 0.0f);
    std::vector<float> full_recurrent_state(total_recurrent_state, 0.0f);

    terlml::hgrn_model_f32(
        token_ids.data(),
        weights,
        {
            .conv_cache = initial_conv_state.data(),
            .recurrent_state = initial_recurrent_state.data(),
        },
        full_logits.data(),
        {
            .conv_cache = full_conv_state.data(),
            .recurrent_state = full_recurrent_state.data(),
        },
        epsilon,
        shape
    );

    std::vector<float> manual_current(hidden_elements, 0.0f);
    std::vector<float> manual_next(hidden_elements, 0.0f);
    std::vector<float> manual_lower_bounds(layers * hidden, 0.0f);
    std::vector<float> manual_conv_state(total_conv_state, 0.0f);
    std::vector<float> manual_recurrent_state(total_recurrent_state, 0.0f);
    std::vector<float> manual_normalized(hidden_elements, 0.0f);
    std::vector<float> manual_logits(rows * vocab, 0.0f);

    for (std::size_t row = 0; row < rows; ++row) {
        const float * embedding =
            embedding_weight.data() + token_ids[row] * hidden;

        std::copy(
            embedding,
            embedding + hidden,
            manual_current.data() + row * hidden
        );
    }

    terlml::hgrn_lower_bounds_f32(
        raw_lower_bounds.data(),
        manual_lower_bounds.data(),
        layers,
        hidden
    );

    for (std::size_t layer = 0; layer < layers; ++layer) {
        terlml::hgrn_decoder_layer_f32(
            manual_current.data(),
            decoder_layers[layer],
            {
                .conv_cache =
                    initial_conv_state.data() + layer * conv_state_per_layer,
                .recurrent_state =
                    initial_recurrent_state.data() +
                    layer * recurrent_state_per_layer,
            },
            manual_next.data(),
            {
                .conv_cache =
                    manual_conv_state.data() + layer * conv_state_per_layer,
                .recurrent_state =
                    manual_recurrent_state.data() +
                    layer * recurrent_state_per_layer,
            },
            epsilon,
            {
                .batch = batch,
                .sequence = sequence,
                .hidden = hidden,
                .heads = heads,
                .intermediate = intermediate,
                .conv_kernel_size = kernel_size,
            },
            manual_lower_bounds.data() + layer * hidden
        );

        manual_current.swap(manual_next);
    }

    terlml::rmsnorm_f32(
        manual_current.data(),
        final_norm_weight.data(),
        epsilon,
        manual_normalized.data(),
        {
            .rows = rows,
            .hidden = hidden,
        }
    );

    terlml::fused_bitlinear_f32(
        manual_normalized.data(),
        lm_head_norm_weight.data(),
        lm_head_weight.data(),
        manual_logits.data(),
        epsilon,
        {
            .rows = rows,
            .input_features = hidden,
            .output_features = vocab,
        }
    );

    const std::vector<std::uint32_t> first_ids = slice_ids(
        token_ids,
        0,
        first_sequence
    );
    const std::vector<std::uint32_t> second_ids = slice_ids(
        token_ids,
        first_sequence,
        second_sequence
    );

    std::vector<float> first_logits(first_sequence * vocab, 0.0f);
    std::vector<float> first_conv_state(total_conv_state, 0.0f);
    std::vector<float> first_recurrent_state(total_recurrent_state, 0.0f);

    terlml::hgrn_model_f32(
        first_ids.data(),
        weights,
        {
            .conv_cache = initial_conv_state.data(),
            .recurrent_state = initial_recurrent_state.data(),
        },
        first_logits.data(),
        {
            .conv_cache = first_conv_state.data(),
            .recurrent_state = first_recurrent_state.data(),
        },
        epsilon,
        {
            .batch = batch,
            .sequence = first_sequence,
            .vocab = vocab,
            .layers = layers,
            .hidden = hidden,
            .heads = heads,
            .intermediate = intermediate,
            .conv_kernel_size = kernel_size,
        }
    );

    std::vector<float> second_logits(second_sequence * vocab, 0.0f);
    std::vector<float> second_conv_state(total_conv_state, 0.0f);
    std::vector<float> second_recurrent_state(total_recurrent_state, 0.0f);

    terlml::hgrn_model_f32(
        second_ids.data(),
        weights,
        {
            .conv_cache = first_conv_state.data(),
            .recurrent_state = first_recurrent_state.data(),
        },
        second_logits.data(),
        {
            .conv_cache = second_conv_state.data(),
            .recurrent_state = second_recurrent_state.data(),
        },
        epsilon,
        {
            .batch = batch,
            .sequence = second_sequence,
            .vocab = vocab,
            .layers = layers,
            .hidden = hidden,
            .heads = heads,
            .intermediate = intermediate,
            .conv_kernel_size = kernel_size,
        }
    );

    const std::vector<float> split_logits = join_rows(
        first_logits,
        second_logits
    );

    return expect_close(
               "model wrapper versus manual composition logits",
               full_logits,
               manual_logits
           ) &&
           expect_close(
               "model wrapper versus manual convolution state",
               full_conv_state,
               manual_conv_state
           ) &&
           expect_close(
               "model wrapper versus manual recurrent state",
               full_recurrent_state,
               manual_recurrent_state
           ) &&
           expect_close(
               "model prefill versus continuation logits",
               split_logits,
               full_logits
           ) &&
           expect_close(
               "model prefill versus continuation convolution state",
               second_conv_state,
               full_conv_state
           ) &&
           expect_close(
               "model prefill versus continuation recurrent state",
               second_recurrent_state,
               full_recurrent_state
           );
}

} // namespace

int main() {
    if (!test_model_composition_and_continuation()) {
        std::cerr << "\nHGRN model tests failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nAll HGRN model tests passed.\n";
    return EXIT_SUCCESS;
}
