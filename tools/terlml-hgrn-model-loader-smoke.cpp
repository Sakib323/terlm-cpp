#include "../terlml/terlml-hgrn-model-loader.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " MODEL.safetensors\n";
        return EXIT_FAILURE;
    }

    try {
        const auto started = std::chrono::steady_clock::now();

        const terlml::hgrn_model_storage storage =
            terlml::hgrn_model_storage::load_f16_safetensors(argv[1]);

        const auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started
        ).count();

        const terlml::hgrn_model_shape & shape = storage.shape();
        const terlml::hgrn_model_weights & weights = storage.weights();

        if (weights.embedding_weight == nullptr ||
            weights.decoder_layers == nullptr ||
            weights.raw_lower_bounds == nullptr ||
            weights.final_norm_weight == nullptr ||
            weights.lm_head_norm_weight == nullptr ||
            weights.lm_head_weight == nullptr) {
            throw std::runtime_error("loader returned a null global weight view");
        }

        if (weights.decoder_layers[0].attention.conv_bias != nullptr) {
            throw std::runtime_error("checkpoint unexpectedly has convolution bias");
        }

        if (weights.decoder_layers[0].attention.conv_weight == nullptr) {
            throw std::runtime_error("layer 0 convolution weight is null");
        }

        std::cout
            << "Loaded TerLM HGRN checkpoint\n"
            << "vocab=" << shape.vocab
            << " layers=" << shape.layers
            << " hidden=" << shape.hidden
            << " heads=" << shape.heads
            << " intermediate=" << shape.intermediate
            << " conv_kernel_size=" << shape.conv_kernel_size
            << '\n'
            << "layer0_conv_bias=null\n"
            << "load_seconds=" << elapsed << '\n';

        return EXIT_SUCCESS;
    } catch (const std::exception & error) {
        std::cerr << "load failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
