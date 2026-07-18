#include "../terlml/terlml-hgrn-model-loader.h"

#include "../terlml/terlml-hgrn-model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " MODEL.safetensors\n";
        return EXIT_FAILURE;
    }

    try {
        const auto load_started = std::chrono::steady_clock::now();

        const terlml::hgrn_model_storage storage =
            terlml::hgrn_model_storage::load_f16_safetensors(argv[1]);

        const double load_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - load_started
        ).count();

        terlml::hgrn_model_shape shape = storage.shape();
        shape.batch = 1;
        shape.sequence = 1;

        const std::size_t rows = shape.batch * shape.sequence;
        const std::size_t total_conv_state =
            shape.layers *
            shape.batch *
            shape.hidden *
            shape.conv_kernel_size;
        const std::size_t total_recurrent_state =
            shape.layers * shape.batch * shape.hidden;

        const std::vector<std::uint32_t> token_ids = {1};
        std::vector<float> initial_conv_state(total_conv_state, 0.0f);
        std::vector<float> initial_recurrent_state(
            total_recurrent_state,
            0.0f
        );
        std::vector<float> final_conv_state(total_conv_state, 0.0f);
        std::vector<float> final_recurrent_state(
            total_recurrent_state,
            0.0f
        );
        std::vector<float> logits(rows * shape.vocab, 0.0f);

        const auto forward_started = std::chrono::steady_clock::now();

        terlml::hgrn_model_f32(
            token_ids.data(),
            storage.weights(),
            {
                .conv_cache = initial_conv_state.data(),
                .recurrent_state = initial_recurrent_state.data(),
            },
            logits.data(),
            {
                .conv_cache = final_conv_state.data(),
                .recurrent_state = final_recurrent_state.data(),
            },
            1e-5f,
            shape
        );

        const double forward_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - forward_started
        ).count();

        const auto non_finite = std::find_if(
            logits.begin(),
            logits.end(),
            [](float value) {
                return !std::isfinite(value);
            }
        );

        if (non_finite != logits.end()) {
            throw std::runtime_error(
                "non-finite logit at index " +
                std::to_string(
                    static_cast<std::size_t>(non_finite - logits.begin())
                )
            );
        }

        const auto argmax = std::max_element(logits.begin(), logits.end());
        const std::size_t token_id = static_cast<std::size_t>(
            argmax - logits.begin()
        );

        std::cout
            << "Loaded and evaluated TerLM HGRN checkpoint\n"
            << "input_token_id=1\n"
            << "vocab=" << shape.vocab
            << " layers=" << shape.layers
            << " hidden=" << shape.hidden
            << " heads=" << shape.heads
            << " intermediate=" << shape.intermediate
            << " conv_kernel_size=" << shape.conv_kernel_size
            << '\n'
            << "conv_state_floats=" << total_conv_state
            << " recurrent_state_floats=" << total_recurrent_state
            << '\n'
            << "argmax_token_id=" << token_id
            << " argmax_logit=" << *argmax
            << '\n'
            << "load_seconds=" << load_seconds
            << " forward_seconds=" << forward_seconds
            << '\n';

        return EXIT_SUCCESS;
    } catch (const std::exception & error) {
        std::cerr << "forward smoke failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
