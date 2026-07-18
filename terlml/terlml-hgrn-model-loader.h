#pragma once

#include "terlml-hgrn-model.h"

#include <filesystem>
#include <vector>

namespace terlml {

class hgrn_model_storage {
public:
    static hgrn_model_storage
    load_f16_safetensors(const std::filesystem::path & path);

    hgrn_model_storage(const hgrn_model_storage &) = delete;
    hgrn_model_storage & operator=(const hgrn_model_storage &) = delete;
    hgrn_model_storage(hgrn_model_storage &&) = delete;
    hgrn_model_storage & operator=(hgrn_model_storage &&) = delete;

    const hgrn_model_shape & shape() const;
    const hgrn_model_weights & weights() const;

private:
    struct decoder_layer_storage {
        std::vector<float> conv_weight;

        std::vector<float> i_proj_norm_weight;
        std::vector<float> i_proj_weight;

        std::vector<float> f_proj_norm_weight;
        std::vector<float> f_proj_weight;

        std::vector<float> g_proj_norm_weight;
        std::vector<float> g_proj_weight;
        std::vector<float> gnorm_weight;

        std::vector<float> o_proj_norm_weight;
        std::vector<float> o_proj_weight;

        std::vector<float> gate_proj_norm_weight;
        std::vector<float> gate_proj_weight;

        std::vector<float> down_proj_norm_weight;
        std::vector<float> down_proj_weight;
    };

    explicit hgrn_model_storage(const std::filesystem::path & path);

    void rebuild_weight_view();

    std::vector<float> embedding_weight_;
    std::vector<float> raw_lower_bounds_;
    std::vector<float> final_norm_weight_;
    std::vector<float> lm_head_norm_weight_;
    std::vector<float> lm_head_weight_;

    std::vector<decoder_layer_storage> layers_;
    std::vector<hgrn_decoder_layer_weights> layer_weight_views_;

    hgrn_model_shape shape_ = {
        .batch = 1,
        .sequence = 1,
        .vocab = 32000,
        .layers = 32,
        .hidden = 1536,
        .heads = 16,
        .intermediate = 4096,
        .conv_kernel_size = 4,
    };

    hgrn_model_weights weight_view_ = {};
};

} // namespace terlml
