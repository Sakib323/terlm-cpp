#include "terlml-hgrn-model-loader.h"

#include "terlml-safetensors.h"

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace terlml {

namespace {

std::vector<float> read_required_f16(
    const safetensors_file & file,
    const std::string & name,
    std::initializer_list<std::size_t> expected_shape
) {
    const safetensors_tensor_info & info = file.tensor_info(name);
    const std::vector<std::size_t> expected(expected_shape);

    if (info.dtype != "F16") {
        throw std::runtime_error(
            "TerLM loader: tensor '" + name + "' must have dtype F16"
        );
    }

    if (info.shape != expected) {
        throw std::runtime_error(
            "TerLM loader: unexpected shape for tensor '" + name + "'"
        );
    }

    return file.read_f16_as_f32(name);
}

std::string layer_prefix(std::size_t layer) {
    return "model.layers." + std::to_string(layer);
}

} // namespace

hgrn_model_storage
hgrn_model_storage::load_f16_safetensors(const std::filesystem::path & path) {
    return hgrn_model_storage(path);
}

hgrn_model_storage::hgrn_model_storage(
    const std::filesystem::path & path
) {
    const safetensors_file file(path);

    embedding_weight_ = read_required_f16(
        file,
        "model.embeddings.weight",
        {shape_.vocab, shape_.hidden}
    );

    raw_lower_bounds_ = read_required_f16(
        file,
        "model.lower_bounds",
        {shape_.layers, shape_.hidden}
    );

    final_norm_weight_ = read_required_f16(
        file,
        "model.norm.weight",
        {shape_.hidden}
    );

    lm_head_norm_weight_ = read_required_f16(
        file,
        "lm_head.norm.weight",
        {shape_.hidden}
    );

    lm_head_weight_ = read_required_f16(
        file,
        "lm_head.weight",
        {shape_.vocab, shape_.hidden}
    );

    layers_.resize(shape_.layers);

    for (std::size_t layer = 0; layer < shape_.layers; ++layer) {
        decoder_layer_storage & storage = layers_[layer];
        const std::string prefix = layer_prefix(layer);

        storage.f_proj_norm_weight = read_required_f16(
            file,
            prefix + ".attn.f_proj.norm.weight",
            {shape_.hidden}
        );
        storage.f_proj_weight = read_required_f16(
            file,
            prefix + ".attn.f_proj.weight",
            {shape_.hidden, shape_.hidden}
        );

        storage.gnorm_weight = read_required_f16(
            file,
            prefix + ".attn.g_norm.weight",
            {shape_.hidden}
        );

        storage.g_proj_norm_weight = read_required_f16(
            file,
            prefix + ".attn.g_proj.norm.weight",
            {shape_.hidden}
        );
        storage.g_proj_weight = read_required_f16(
            file,
            prefix + ".attn.g_proj.weight",
            {shape_.hidden, shape_.hidden}
        );

        storage.conv_weight = read_required_f16(
            file,
            prefix + ".attn.h_conv1d.weight",
            {shape_.hidden, 1, shape_.conv_kernel_size}
        );

        storage.i_proj_norm_weight = read_required_f16(
            file,
            prefix + ".attn.i_proj.norm.weight",
            {shape_.hidden}
        );
        storage.i_proj_weight = read_required_f16(
            file,
            prefix + ".attn.i_proj.weight",
            {shape_.hidden, shape_.hidden}
        );

        storage.o_proj_norm_weight = read_required_f16(
            file,
            prefix + ".attn.o_proj.norm.weight",
            {shape_.hidden}
        );
        storage.o_proj_weight = read_required_f16(
            file,
            prefix + ".attn.o_proj.weight",
            {shape_.hidden, shape_.hidden}
        );

        storage.down_proj_norm_weight = read_required_f16(
            file,
            prefix + ".mlp.down_proj.norm.weight",
            {shape_.intermediate}
        );
        storage.down_proj_weight = read_required_f16(
            file,
            prefix + ".mlp.down_proj.weight",
            {shape_.hidden, shape_.intermediate}
        );

        storage.gate_proj_norm_weight = read_required_f16(
            file,
            prefix + ".mlp.gate_proj.norm.weight",
            {shape_.hidden}
        );
        storage.gate_proj_weight = read_required_f16(
            file,
            prefix + ".mlp.gate_proj.weight",
            {2 * shape_.intermediate, shape_.hidden}
        );
    }

    rebuild_weight_view();
}

void hgrn_model_storage::rebuild_weight_view() {
    layer_weight_views_.resize(layers_.size());

    for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
        const decoder_layer_storage & storage = layers_[layer];

        layer_weight_views_[layer] = {
            .attention = {
                .conv_weight = storage.conv_weight.data(),
                .conv_bias = nullptr,
                .i_proj_norm_weight = storage.i_proj_norm_weight.data(),
                .i_proj_weight = storage.i_proj_weight.data(),
                .f_proj_norm_weight = storage.f_proj_norm_weight.data(),
                .f_proj_weight = storage.f_proj_weight.data(),
                .g_proj_norm_weight = storage.g_proj_norm_weight.data(),
                .g_proj_weight = storage.g_proj_weight.data(),
                .gnorm_weight = storage.gnorm_weight.data(),
                .o_proj_norm_weight = storage.o_proj_norm_weight.data(),
                .o_proj_weight = storage.o_proj_weight.data(),
            },
            .mlp = {
                .gate_proj_norm_weight = storage.gate_proj_norm_weight.data(),
                .gate_proj_weight = storage.gate_proj_weight.data(),
                .down_proj_norm_weight = storage.down_proj_norm_weight.data(),
                .down_proj_weight = storage.down_proj_weight.data(),
            },
        };
    }

    weight_view_ = {
        .embedding_weight = embedding_weight_.data(),
        .decoder_layers = layer_weight_views_.data(),
        .raw_lower_bounds = raw_lower_bounds_.data(),
        .final_norm_weight = final_norm_weight_.data(),
        .lm_head_norm_weight = lm_head_norm_weight_.data(),
        .lm_head_weight = lm_head_weight_.data(),
    };
}

const hgrn_model_shape & hgrn_model_storage::shape() const {
    return shape_;
}

const hgrn_model_weights & hgrn_model_storage::weights() const {
    return weight_view_;
}

} // namespace terlml
