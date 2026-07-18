#include "../terlml/terlml-causal-conv.h"
#include "../terlml/terlml-fused-bitlinear.h"
#include "../terlml/terlml-hgrn-attention-block.h"
#include "../terlml/terlml-hgrn-model-loader.h"
#include "../terlml/terlml-rmsnorm-gate.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

constexpr std::array<char, 8> k_magic = {
    'T', 'E', 'R', 'L', 'M', 'D', '0', '\0'
};

struct fixture {
    std::vector<float> convolved;
    std::vector<float> projected_i;
    std::vector<float> projected_f;
    std::vector<float> projected_g;
    std::vector<float> g_norm_gate;
    std::vector<float> recurrent_output;
    std::vector<float> gated_output;
    std::vector<float> attention_output;
    std::vector<float> conv_state;
    std::vector<float> recurrent_state;
};

template <typename T>
bool read_value(std::ifstream & stream, T * value, const char * name) {
    stream.read(
        reinterpret_cast<char *>(value),
        static_cast<std::streamsize>(sizeof(T))
    );

    if (!stream) {
        std::cerr << "[FAIL] read " << name << '\n';
        return false;
    }

    return true;
}

bool read_floats(
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
        std::cerr << "[FAIL] read " << name << '\n';
        return false;
    }

    return true;
}

bool read_fixture(const std::filesystem::path & path, fixture * output) {
    std::ifstream stream(path, std::ios::binary);

    if (!stream) {
        std::cerr << "[FAIL] open fixture: " << path << '\n';
        return false;
    }

    std::array<char, 8> magic{};
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));

    if (!stream || magic != k_magic) {
        std::cerr << "[FAIL] invalid TERLMD0 fixture magic\n";
        return false;
    }

    std::uint32_t version = 0;
    std::uint32_t hidden = 0;
    std::uint32_t kernel = 0;
    std::uint32_t tensors = 0;

    if (!read_value(stream, &version, "version") ||
        !read_value(stream, &hidden, "hidden") ||
        !read_value(stream, &kernel, "kernel") ||
        !read_value(stream, &tensors, "tensor count")) {
        return false;
    }

    if (version != 3 || hidden != 1536 || kernel != 4 || tensors != 10) {
        std::cerr
            << "[FAIL] unexpected header:"
            << " version=" << version
            << " hidden=" << hidden
            << " kernel=" << kernel
            << " tensors=" << tensors
            << '\n';
        return false;
    }

    if (!read_floats(stream, &output->convolved, hidden, "convolved") ||
        !read_floats(stream, &output->projected_i, hidden, "projected_i") ||
        !read_floats(stream, &output->projected_f, hidden, "projected_f") ||
        !read_floats(stream, &output->projected_g, hidden, "projected_g") ||
        !read_floats(
            stream,
            &output->g_norm_gate,
            hidden,
            "g_norm_gate"
        ) ||
        !read_floats(
            stream,
            &output->recurrent_output,
            hidden,
            "recurrent_output"
        ) ||
        !read_floats(
            stream,
            &output->gated_output,
            hidden,
            "gated_output"
        ) ||
        !read_floats(
            stream,
            &output->attention_output,
            hidden,
            "attention_output"
        ) ||
        !read_floats(
            stream,
            &output->conv_state,
            hidden * kernel,
            "conv_state"
        ) ||
        !read_floats(
            stream,
            &output->recurrent_state,
            hidden,
            "recurrent_state"
        )) {
        return false;
    }

    char trailing = '\0';
    stream.read(&trailing, 1);

    if (stream.gcount() != 0) {
        std::cerr << "[FAIL] fixture has trailing bytes\n";
        return false;
    }

    return true;
}

void compare(
    const char * name,
    const std::vector<float> & actual,
    const std::vector<float> & expected
) {
    float maximum = 0.0f;
    std::size_t index = 0;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float error = std::fabs(actual[i] - expected[i]);

        if (error > maximum) {
            maximum = error;
            index = i;
        }
    }

    std::cout
        << name
        << ": max_abs=" << maximum
        << " index=" << index
        << " cpp=" << actual[index]
        << " python=" << expected[index]
        << '\n';
}

bool run(
    const std::filesystem::path & model_path,
    const std::filesystem::path & fixture_path
) {
    fixture expected{};

    if (!read_fixture(fixture_path, &expected)) {
        return false;
    }

    const terlml::hgrn_model_storage storage =
        terlml::hgrn_model_storage::load_f16_safetensors(model_path);

    terlml::hgrn_model_shape shape = storage.shape();
    shape.batch = 1;
    shape.sequence = 1;

    if (shape.hidden != 1536 || shape.conv_kernel_size != 4) {
        std::cerr << "[FAIL] checkpoint shape does not match fixture\n";
        return false;
    }

    const auto weights = storage.weights();
    const auto & attention = weights.decoder_layers[0].attention;

    const std::size_t hidden = shape.hidden;
    const std::size_t conv_size = hidden * shape.conv_kernel_size;

    const float * embedding =
        weights.embedding_weight + static_cast<std::size_t>(1) * hidden;

    std::vector<float> input(embedding, embedding + hidden);
    std::vector<float> zeros_conv(conv_size, 0.0f);
    std::vector<float> zeros_recurrent(hidden, 0.0f);

    std::vector<float> convolved(hidden, 0.0f);
    std::vector<float> conv_state(conv_size, 0.0f);
    std::vector<float> projected_i(hidden, 0.0f);
    std::vector<float> projected_f(hidden, 0.0f);
    std::vector<float> projected_g(hidden, 0.0f);
    std::vector<float> recurrent_input(hidden, 0.0f);
    std::vector<float> gated_output(hidden, 0.0f);
    std::vector<float> gated_output_from_python_inputs(hidden, 0.0f);
    std::vector<float> attention_output(hidden, 0.0f);
    std::vector<float> final_conv(conv_size, 0.0f);
    std::vector<float> final_recurrent(hidden, 0.0f);

    terlml::causal_depthwise_conv1d_silu_f32(
        input.data(),
        attention.conv_weight,
        attention.conv_bias,
        zeros_conv.data(),
        convolved.data(),
        conv_state.data(),
        {
            .batch = 1,
            .sequence = 1,
            .channels = hidden,
            .kernel_size = shape.conv_kernel_size,
        }
    );

    const terlml::fused_bitlinear_shape projection_shape = {
        .rows = 1,
        .input_features = hidden,
        .output_features = hidden,
    };

    terlml::fused_bitlinear_f32(
        convolved.data(),
        attention.i_proj_norm_weight,
        attention.i_proj_weight,
        projected_i.data(),
        1e-6f,
        projection_shape
    );

    terlml::fused_bitlinear_f32(
        convolved.data(),
        attention.f_proj_norm_weight,
        attention.f_proj_weight,
        projected_f.data(),
        1e-6f,
        projection_shape
    );

    terlml::fused_bitlinear_f32(
        convolved.data(),
        attention.g_proj_norm_weight,
        attention.g_proj_weight,
        projected_g.data(),
        1e-6f,
        projection_shape
    );    for (std::size_t index = 0; index < hidden; ++index) {
        const float decay =
            1.0f / (1.0f + std::exp(-projected_f[index]));

        const float input_gate =
            projected_i[index] /
            (1.0f + std::exp(-projected_i[index]));

        recurrent_input[index] = input_gate * (1.0f - decay);
    }

    terlml::rmsnorm_silu_gate_f32(
        projected_g.data(),
        recurrent_input.data(),
        attention.gnorm_weight,
        1e-6f,
        gated_output.data(),
        {
            .batch = 1,
            .sequence = 1,
            .hidden = hidden,
        }
    );    terlml::rmsnorm_silu_gate_f32(
        expected.g_norm_gate.data(),
        expected.recurrent_output.data(),
        attention.gnorm_weight,
        1e-5f,
        gated_output_from_python_inputs.data(),
        {
            .batch = 1,
            .sequence = 1,
            .hidden = hidden,
        }
    );





    terlml::hgrn_attention_block_f32(
        input.data(),
        attention,
        {
            .conv_cache = zeros_conv.data(),
            .recurrent_state = zeros_recurrent.data(),
        },
        attention_output.data(),
        {
            .conv_cache = final_conv.data(),
            .recurrent_state = final_recurrent.data(),
        },
        1e-6f,
        {
            .batch = 1,
            .sequence = 1,
            .hidden = hidden,
            .heads = shape.heads,
            .conv_kernel_size = shape.conv_kernel_size,
        }
    );

    compare("convolved", convolved, expected.convolved);
    compare("conv_state", conv_state, expected.conv_state);
    compare("projected_i", projected_i, expected.projected_i);
    compare("projected_f", projected_f, expected.projected_f);
    compare("projected_g", projected_g, expected.projected_g);
    compare("g_norm_gate", projected_g, expected.g_norm_gate);
    compare("recurrent_output", recurrent_input, expected.recurrent_output);
    compare("gated_output", gated_output, expected.gated_output);
    compare(
        "gated_output_exact_inputs",
        gated_output_from_python_inputs,
        expected.gated_output
    );
    compare("attention_output", attention_output, expected.attention_output);
    compare("final_conv_state", final_conv, expected.conv_state);
    compare("recurrent_state", final_recurrent, expected.recurrent_state);

    return true;
}

} // namespace

int main() {
    const char * model_path = std::getenv("TERLML_TEST_MODEL_PATH");
    const char * fixture_path = std::getenv("TERLML_LAYER0_DEBUG_FIXTURE_PATH");

    if (model_path == nullptr || fixture_path == nullptr) {
        std::cerr
            << "Set TERLML_TEST_MODEL_PATH and "
            << "TERLML_LAYER0_DEBUG_FIXTURE_PATH\n";
        return EXIT_FAILURE;
    }

    return run(model_path, fixture_path) ? EXIT_SUCCESS : EXIT_FAILURE;
}
