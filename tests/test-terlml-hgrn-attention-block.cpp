#include "../terlml/terlml-hgrn-attention-block.h"

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

std::vector<float> slice_tokens(
    const std::vector<float> & source,
    std::size_t batch,
    std::size_t full_sequence,
    std::size_t hidden,
    std::size_t start,
    std::size_t count
) {
    std::vector<float> result(batch * count * hidden);

    for (std::size_t b = 0; b < batch; ++b) {
        const float * source_begin =
            source.data() + (b * full_sequence + start) * hidden;

        float * destination_begin =
            result.data() + b * count * hidden;

        std::copy(
            source_begin,
            source_begin + count * hidden,
            destination_begin
        );
    }

    return result;
}

std::vector<float> join_tokens(
    const std::vector<float> & first,
    const std::vector<float> & second,
    std::size_t batch,
    std::size_t first_sequence,
    std::size_t second_sequence,
    std::size_t hidden
) {
    std::vector<float> result(
        batch * (first_sequence + second_sequence) * hidden
    );

    for (std::size_t b = 0; b < batch; ++b) {
        const float * first_source =
            first.data() + b * first_sequence * hidden;

        const float * second_source =
            second.data() + b * second_sequence * hidden;

        float * destination =
            result.data() +
            b * (first_sequence + second_sequence) * hidden;

        std::copy(
            first_source,
            first_source + first_sequence * hidden,
            destination
        );

        std::copy(
            second_source,
            second_source + second_sequence * hidden,
            destination + first_sequence * hidden
        );
    }

    return result;
}

bool test_prefill_equals_continuation() {
    constexpr std::size_t batch = 2;
    constexpr std::size_t sequence = 5;
    constexpr std::size_t first_sequence = 2;
    constexpr std::size_t second_sequence = sequence - first_sequence;
    constexpr std::size_t hidden = 12;
    constexpr std::size_t heads = 3;
    constexpr std::size_t kernel_size = 3;
    constexpr float epsilon = 1e-5f;

    const std::size_t input_elements = batch * sequence * hidden;
    const std::size_t conv_state_elements = batch * hidden * kernel_size;
    const std::size_t recurrent_state_elements = batch * hidden;
    const std::size_t projection_elements = hidden * hidden;
    const std::size_t conv_weight_elements = hidden * kernel_size;

    std::vector<float> input(input_elements);
    std::vector<float> conv_weight(conv_weight_elements);
    std::vector<float> conv_bias(hidden);

    std::vector<float> i_proj_norm_weight(hidden);
    std::vector<float> i_proj_weight(projection_elements);

    std::vector<float> f_proj_norm_weight(hidden);
    std::vector<float> f_proj_weight(projection_elements);

    std::vector<float> g_proj_norm_weight(hidden);
    std::vector<float> g_proj_weight(projection_elements);

    std::vector<float> gnorm_weight(hidden);

    std::vector<float> o_proj_norm_weight(hidden);
    std::vector<float> o_proj_weight(projection_elements);

    std::vector<float> initial_conv_state(conv_state_elements);
    std::vector<float> initial_recurrent_state(recurrent_state_elements);

    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = deterministic_value(index, 1.0f);
    }

    for (std::size_t index = 0; index < conv_weight.size(); ++index) {
        conv_weight[index] = deterministic_value(index + 101, 0.2f);
    }

    for (std::size_t index = 0; index < hidden; ++index) {
        conv_bias[index] = deterministic_value(index + 211, 0.1f);
        i_proj_norm_weight[index] =
            0.75f + deterministic_value(index + 307, 0.2f);
        f_proj_norm_weight[index] =
            0.75f + deterministic_value(index + 401, 0.2f);
        g_proj_norm_weight[index] =
            0.75f + deterministic_value(index + 503, 0.2f);
        gnorm_weight[index] =
            0.75f + deterministic_value(index + 607, 0.2f);
        o_proj_norm_weight[index] =
            0.75f + deterministic_value(index + 709, 0.2f);
    }

    for (std::size_t index = 0; index < projection_elements; ++index) {
        i_proj_weight[index] = deterministic_value(index + 809, 0.15f);
        f_proj_weight[index] = deterministic_value(index + 907, 0.15f);
        g_proj_weight[index] = deterministic_value(index + 1009, 0.15f);
        o_proj_weight[index] = deterministic_value(index + 1103, 0.15f);
    }

    for (std::size_t index = 0; index < initial_conv_state.size(); ++index) {
        initial_conv_state[index] = deterministic_value(index + 1201, 0.5f);
    }

    for (std::size_t index = 0;
         index < initial_recurrent_state.size();
         ++index) {
        initial_recurrent_state[index] =
            deterministic_value(index + 1301, 0.5f);
    }

    const terlml::hgrn_attention_block_weights weights = {
        .conv_weight = conv_weight.data(),
        .conv_bias = conv_bias.data(),
        .i_proj_norm_weight = i_proj_norm_weight.data(),
        .i_proj_weight = i_proj_weight.data(),
        .f_proj_norm_weight = f_proj_norm_weight.data(),
        .f_proj_weight = f_proj_weight.data(),
        .g_proj_norm_weight = g_proj_norm_weight.data(),
        .g_proj_weight = g_proj_weight.data(),
        .gnorm_weight = gnorm_weight.data(),
        .o_proj_norm_weight = o_proj_norm_weight.data(),
        .o_proj_weight = o_proj_weight.data(),
    };

    const terlml::hgrn_attention_block_state initial_state = {
        .conv_cache = initial_conv_state.data(),
        .recurrent_state = initial_recurrent_state.data(),
    };

    std::vector<float> prefill_output(input_elements, 0.0f);
    std::vector<float> prefill_conv_state(conv_state_elements, 0.0f);
    std::vector<float> prefill_recurrent_state(recurrent_state_elements, 0.0f);

    terlml::hgrn_attention_block_f32(
        input.data(),
        weights,
        initial_state,
        prefill_output.data(),
        {
            .conv_cache = prefill_conv_state.data(),
            .recurrent_state = prefill_recurrent_state.data(),
        },
        epsilon,
        {
            .batch = batch,
            .sequence = sequence,
            .hidden = hidden,
            .heads = heads,
            .conv_kernel_size = kernel_size,
        }
    );

    const std::vector<float> first_input = slice_tokens(
        input,
        batch,
        sequence,
        hidden,
        0,
        first_sequence
    );

    const std::vector<float> second_input = slice_tokens(
        input,
        batch,
        sequence,
        hidden,
        first_sequence,
        second_sequence
    );

    std::vector<float> first_output(
        batch * first_sequence * hidden,
        0.0f
    );
    std::vector<float> first_conv_state(conv_state_elements, 0.0f);
    std::vector<float> first_recurrent_state(recurrent_state_elements, 0.0f);

    terlml::hgrn_attention_block_f32(
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
            .conv_kernel_size = kernel_size,
        }
    );

    std::vector<float> second_output(
        batch * second_sequence * hidden,
        0.0f
    );
    std::vector<float> second_conv_state(conv_state_elements, 0.0f);
    std::vector<float> second_recurrent_state(recurrent_state_elements, 0.0f);

    terlml::hgrn_attention_block_f32(
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
            .conv_kernel_size = kernel_size,
        }
    );

    const std::vector<float> split_output = join_tokens(
        first_output,
        second_output,
        batch,
        first_sequence,
        second_sequence,
        hidden
    );

    return expect_close(
               "prefill versus split output",
               split_output,
               prefill_output
           ) &&
           expect_close(
               "prefill versus split convolution cache",
               second_conv_state,
               prefill_conv_state
           ) &&
           expect_close(
               "prefill versus split recurrent state",
               second_recurrent_state,
               prefill_recurrent_state
           );
}

} // namespace

int main() {
    if (!test_prefill_equals_continuation()) {
        std::cerr << "\nHGRN attention-block tests failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nAll HGRN attention-block tests passed.\n";
    return EXIT_SUCCESS;
}
