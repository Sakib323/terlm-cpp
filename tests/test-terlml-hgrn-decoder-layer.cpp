#include "../terlml/terlml-hgrn-attention-block.h"
#include "../terlml/terlml-hgrn-decoder-layer.h"
#include "../terlml/terlml-hgrn-mlp.h"

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

bool test_decoder_layer() {
    constexpr std::size_t batch = 1;
    constexpr std::size_t sequence = 5;
    constexpr std::size_t first_sequence = 2;
    constexpr std::size_t second_sequence = sequence - first_sequence;
    constexpr std::size_t hidden = 12;
    constexpr std::size_t heads = 3;
    constexpr std::size_t intermediate = 20;
    constexpr std::size_t kernel_size = 3;
    constexpr float epsilon = 1e-5f;

    const std::size_t elements = batch * sequence * hidden;
    const std::size_t conv_state_elements = batch * hidden * kernel_size;
    const std::size_t recurrent_state_elements = batch * hidden;
    const std::size_t projection_elements = hidden * hidden;
    const std::size_t gate_projection_elements = 2 * intermediate * hidden;
    const std::size_t down_projection_elements = hidden * intermediate;

    std::vector<float> input(elements);
    std::vector<float> lower_bound(hidden);
    std::vector<float> conv_weight(hidden * kernel_size);
    std::vector<float> conv_bias(hidden);

    std::vector<float> i_norm(hidden);
    std::vector<float> i_weight(projection_elements);
    std::vector<float> f_norm(hidden);
    std::vector<float> f_weight(projection_elements);
    std::vector<float> g_norm(hidden);
    std::vector<float> g_weight(projection_elements);
    std::vector<float> gate_norm(hidden);
    std::vector<float> o_norm(hidden);
    std::vector<float> o_weight(projection_elements);

    std::vector<float> mlp_gate_norm(hidden);
    std::vector<float> mlp_gate_weight(gate_projection_elements);
    std::vector<float> mlp_down_norm(intermediate);
    std::vector<float> mlp_down_weight(down_projection_elements);

    std::vector<float> initial_conv_state(conv_state_elements);
    std::vector<float> initial_recurrent_state(recurrent_state_elements);

    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = deterministic_value(index, 0.7f);
    }

    for (std::size_t index = 0; index < hidden; ++index) {
        lower_bound[index] = 0.05f +
            static_cast<float>(index % 5) * 0.03f;
        conv_bias[index] = deterministic_value(index + 101, 0.1f);
        i_norm[index] = 0.8f + deterministic_value(index + 201, 0.15f);
        f_norm[index] = 0.8f + deterministic_value(index + 301, 0.15f);
        g_norm[index] = 0.8f + deterministic_value(index + 401, 0.15f);
        gate_norm[index] = 0.8f + deterministic_value(index + 501, 0.15f);
        o_norm[index] = 0.8f + deterministic_value(index + 601, 0.15f);
        mlp_gate_norm[index] =
            0.8f + deterministic_value(index + 701, 0.15f);
    }

    for (std::size_t index = 0; index < intermediate; ++index) {
        mlp_down_norm[index] =
            0.8f + deterministic_value(index + 801, 0.15f);
    }

    for (std::size_t index = 0; index < conv_weight.size(); ++index) {
        conv_weight[index] = deterministic_value(index + 901, 0.2f);
    }

    for (std::size_t index = 0; index < projection_elements; ++index) {
        i_weight[index] = deterministic_value(index + 1001, 0.15f);
        f_weight[index] = deterministic_value(index + 2001, 0.15f);
        g_weight[index] = deterministic_value(index + 3001, 0.15f);
        o_weight[index] = deterministic_value(index + 4001, 0.15f);
    }

    for (std::size_t index = 0;
         index < mlp_gate_weight.size();
         ++index) {
        mlp_gate_weight[index] =
            deterministic_value(index + 5001, 0.15f);
    }

    for (std::size_t index = 0;
         index < mlp_down_weight.size();
         ++index) {
        mlp_down_weight[index] =
            deterministic_value(index + 6001, 0.15f);
    }

    for (std::size_t index = 0;
         index < initial_conv_state.size();
         ++index) {
        initial_conv_state[index] =
            deterministic_value(index + 7001, 0.4f);
    }

    for (std::size_t index = 0;
         index < initial_recurrent_state.size();
         ++index) {
        initial_recurrent_state[index] =
            deterministic_value(index + 8001, 0.4f);
    }

    const terlml::hgrn_decoder_layer_weights weights = {
        .attention = {
            .conv_weight = conv_weight.data(),
            .conv_bias = conv_bias.data(),
            .i_proj_norm_weight = i_norm.data(),
            .i_proj_weight = i_weight.data(),
            .f_proj_norm_weight = f_norm.data(),
            .f_proj_weight = f_weight.data(),
            .g_proj_norm_weight = g_norm.data(),
            .g_proj_weight = g_weight.data(),
            .gnorm_weight = gate_norm.data(),
            .o_proj_norm_weight = o_norm.data(),
            .o_proj_weight = o_weight.data(),
        },
        .mlp = {
            .gate_proj_norm_weight = mlp_gate_norm.data(),
            .gate_proj_weight = mlp_gate_weight.data(),
            .down_proj_norm_weight = mlp_down_norm.data(),
            .down_proj_weight = mlp_down_weight.data(),
        },
    };

    const terlml::hgrn_decoder_layer_shape shape = {
        .batch = batch,
        .sequence = sequence,
        .hidden = hidden,
        .heads = heads,
        .intermediate = intermediate,
        .conv_kernel_size = kernel_size,
    };

    const terlml::hgrn_attention_block_state initial_state = {
        .conv_cache = initial_conv_state.data(),
        .recurrent_state = initial_recurrent_state.data(),
    };

    std::vector<float> actual(elements, 0.0f);
    std::vector<float> actual_conv_state(conv_state_elements, 0.0f);
    std::vector<float> actual_recurrent_state(recurrent_state_elements, 0.0f);

    terlml::hgrn_decoder_layer_f32(
        input.data(),
        weights,
        initial_state,
        actual.data(),
        {
            .conv_cache = actual_conv_state.data(),
            .recurrent_state = actual_recurrent_state.data(),
        },
        epsilon,
        shape,
        lower_bound.data()
    );

    std::vector<float> attention_output(elements, 0.0f);
    std::vector<float> expected_conv_state(conv_state_elements, 0.0f);
    std::vector<float> expected_recurrent_state(recurrent_state_elements, 0.0f);
    std::vector<float> after_attention(elements, 0.0f);
    std::vector<float> mlp_output(elements, 0.0f);
    std::vector<float> expected(elements, 0.0f);

    terlml::hgrn_attention_block_f32(
        input.data(),
        weights.attention,
        initial_state,
        attention_output.data(),
        {
            .conv_cache = expected_conv_state.data(),
            .recurrent_state = expected_recurrent_state.data(),
        },
        epsilon,
        {
            .batch = batch,
            .sequence = sequence,
            .hidden = hidden,
            .heads = heads,
            .conv_kernel_size = kernel_size,
        },
        lower_bound.data()
    );

    for (std::size_t index = 0; index < elements; ++index) {
        after_attention[index] = input[index] + attention_output[index];
    }

    terlml::hgrn_mlp_f32(
        after_attention.data(),
        weights.mlp,
        mlp_output.data(),
        {
            .batch = batch,
            .sequence = sequence,
            .hidden = hidden,
            .intermediate = intermediate,
        }
    );

    for (std::size_t index = 0; index < elements; ++index) {
        expected[index] = after_attention[index] + mlp_output[index];
    }

    std::vector<float> first_input(
        input.begin(),
        input.begin() + first_sequence * hidden
    );
    std::vector<float> second_input(
        input.begin() + first_sequence * hidden,
        input.end()
    );

    std::vector<float> first_output(first_sequence * hidden, 0.0f);
    std::vector<float> first_conv_state(conv_state_elements, 0.0f);
    std::vector<float> first_recurrent_state(recurrent_state_elements, 0.0f);

    terlml::hgrn_decoder_layer_f32(
        first_input.data(),
        weights,
        initial_state,
        first_output.data(),
        {
            .conv_cache = first_conv_state.data(),
            .recurrent_state = first_recurrent_state.data(),
        },
        epsilon,
        {
            .batch = batch,
            .sequence = first_sequence,
            .hidden = hidden,
            .heads = heads,
            .intermediate = intermediate,
            .conv_kernel_size = kernel_size,
        },
        lower_bound.data()
    );

    std::vector<float> second_output(second_sequence * hidden, 0.0f);
    std::vector<float> second_conv_state(conv_state_elements, 0.0f);
    std::vector<float> second_recurrent_state(recurrent_state_elements, 0.0f);

    terlml::hgrn_decoder_layer_f32(
        second_input.data(),
        weights,
        {
            .conv_cache = first_conv_state.data(),
            .recurrent_state = first_recurrent_state.data(),
        },
        second_output.data(),
        {
            .conv_cache = second_conv_state.data(),
            .recurrent_state = second_recurrent_state.data(),
        },
        epsilon,
        {
            .batch = batch,
            .sequence = second_sequence,
            .hidden = hidden,
            .heads = heads,
            .intermediate = intermediate,
            .conv_kernel_size = kernel_size,
        },
        lower_bound.data()
    );

    std::vector<float> split_output;
    split_output.reserve(elements);
    split_output.insert(
        split_output.end(),
        first_output.begin(),
        first_output.end()
    );
    split_output.insert(
        split_output.end(),
        second_output.begin(),
        second_output.end()
    );

    return expect_close("decoder wrapper output", actual, expected) &&
           expect_close(
               "decoder convolution state",
               actual_conv_state,
               expected_conv_state
           ) &&
           expect_close(
               "decoder recurrent state",
               actual_recurrent_state,
               expected_recurrent_state
           ) &&
           expect_close(
               "decoder prefill versus continuation output",
               split_output,
               actual
           ) &&
           expect_close(
               "decoder prefill versus continuation convolution state",
               second_conv_state,
               actual_conv_state
           ) &&
           expect_close(
               "decoder prefill versus continuation recurrent state",
               second_recurrent_state,
               actual_recurrent_state
           );
}

} // namespace

int main() {
    if (!test_decoder_layer()) {
        std::cerr << "\nHGRN decoder-layer tests failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nAll HGRN decoder-layer tests passed.\n";
    return EXIT_SUCCESS;
}
