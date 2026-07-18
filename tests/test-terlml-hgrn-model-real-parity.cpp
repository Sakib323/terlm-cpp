#include "../terlml/terlml-hgrn-model-loader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr std::array<char, 8> k_magic = {
    'T', 'E', 'R', 'L', 'M', 'P', 'A', 'R'
};

constexpr std::uint32_t k_version = 1;
constexpr std::uint32_t k_expected_token_id = 1;
constexpr std::uint32_t k_expected_vocab = 32000;
constexpr std::uint32_t k_expected_top_k = 10;
constexpr std::uint32_t k_expected_conv_floats = 196608;
constexpr std::uint32_t k_expected_recurrent_floats = 49152;

struct parity_fixture {
    std::uint32_t token_id = 0;
    std::vector<float> logits;
    std::vector<std::uint32_t> top_ids;
    std::vector<float> top_logits;
    std::vector<float> conv_state;
    std::vector<float> recurrent_state;
};

struct error_summary {
    float maximum_absolute_error = 0.0f;
    float maximum_relative_error = 0.0f;
    std::size_t maximum_absolute_index = 0;
    std::size_t maximum_relative_index = 0;
};

struct layer_trace_writer {
    std::ofstream * stream;
};

void write_layer_trace(
    void * context,
    std::size_t layer,
    const float * attention_output,
    const float * conv_cache,
    const float * recurrent_state,
    terlml::hgrn_model_shape shape
) {
    auto * writer = static_cast<layer_trace_writer *>(context);

    if (writer == nullptr || writer->stream == nullptr) {
        return;
    }

    const std::uint32_t layer_id = static_cast<std::uint32_t>(layer);
    const std::size_t attention_count =
        shape.batch * shape.sequence * shape.hidden;
    const std::size_t conv_count =
        shape.batch * shape.hidden * shape.conv_kernel_size;
    const std::size_t recurrent_count = shape.batch * shape.hidden;

    writer->stream->write(
        reinterpret_cast<const char *>(&layer_id),
        static_cast<std::streamsize>(sizeof(layer_id))
    );
    writer->stream->write(
        reinterpret_cast<const char *>(attention_output),
        static_cast<std::streamsize>(attention_count * sizeof(float))
    );
    writer->stream->write(
        reinterpret_cast<const char *>(conv_cache),
        static_cast<std::streamsize>(conv_count * sizeof(float))
    );
    writer->stream->write(
        reinterpret_cast<const char *>(recurrent_state),
        static_cast<std::streamsize>(recurrent_count * sizeof(float))
    );
}

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

template <typename T>
bool read_vector(
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
        std::cerr << "[FAIL] read " << name << '\n';
        return false;
    }

    return true;
}

bool read_fixture(const std::filesystem::path & path, parity_fixture * fixture) {
    std::ifstream stream(path, std::ios::binary);

    if (!stream) {
        std::cerr << "[FAIL] open fixture: " << path << '\n';
        return false;
    }

    std::array<char, k_magic.size()> magic{};
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));

    if (!stream || magic != k_magic) {
        std::cerr << "[FAIL] invalid TERLMPAR fixture magic\n";
        return false;
    }

    std::uint32_t version = 0;
    std::uint32_t logits_count = 0;
    std::uint32_t top_k = 0;
    std::uint32_t conv_count = 0;
    std::uint32_t recurrent_count = 0;

    if (!read_value(stream, &version, "version") ||
        !read_value(stream, &fixture->token_id, "token_id") ||
        !read_value(stream, &logits_count, "logits_count") ||
        !read_value(stream, &top_k, "top_k") ||
        !read_value(stream, &conv_count, "conv_count") ||
        !read_value(stream, &recurrent_count, "recurrent_count")) {
        return false;
    }

    if (version != k_version ||
        fixture->token_id != k_expected_token_id ||
        logits_count != k_expected_vocab ||
        top_k != k_expected_top_k ||
        conv_count != k_expected_conv_floats ||
        recurrent_count != k_expected_recurrent_floats) {
        std::cerr
            << "[FAIL] unexpected fixture header:"
            << " version=" << version
            << " token_id=" << fixture->token_id
            << " logits=" << logits_count
            << " top_k=" << top_k
            << " conv=" << conv_count
            << " recurrent=" << recurrent_count
            << '\n';
        return false;
    }

    if (!read_vector(stream, &fixture->logits, logits_count, "logits") ||
        !read_vector(stream, &fixture->top_ids, top_k, "top_ids") ||
        !read_vector(stream, &fixture->top_logits, top_k, "top_logits") ||
        !read_vector(stream, &fixture->conv_state, conv_count, "conv_state") ||
        !read_vector(
            stream,
            &fixture->recurrent_state,
            recurrent_count,
            "recurrent_state"
        )) {
        return false;
    }

    char trailing_byte = '\0';
    stream.read(&trailing_byte, 1);

    if (stream.gcount() != 0) {
        std::cerr << "[FAIL] fixture has trailing bytes\n";
        return false;
    }

    return true;
}

error_summary summarize_errors(
    const std::vector<float> & actual,
    const std::vector<float> & expected
) {
    error_summary summary{};

    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float absolute_error = std::fabs(actual[index] - expected[index]);
        const float relative_error =
            absolute_error /
            std::max(std::fabs(expected[index]), 1e-6f);

        if (absolute_error > summary.maximum_absolute_error) {
            summary.maximum_absolute_error = absolute_error;
            summary.maximum_absolute_index = index;
        }

        if (relative_error > summary.maximum_relative_error) {
            summary.maximum_relative_error = relative_error;
            summary.maximum_relative_index = index;
        }
    }

    return summary;
}

void print_error_summary(
    const char * name,
    const std::vector<float> & actual,
    const std::vector<float> & expected
) {
    const error_summary summary = summarize_errors(actual, expected);

    std::cout
        << name
        << ": max_abs_error=" << summary.maximum_absolute_error
        << " at_index=" << summary.maximum_absolute_index
        << " cpp=" << actual[summary.maximum_absolute_index]
        << " python=" << expected[summary.maximum_absolute_index]
        << " max_rel_error=" << summary.maximum_relative_error
        << " at_index=" << summary.maximum_relative_index
        << '\n';
}

std::vector<std::size_t> top_indices(
    const std::vector<float> & values,
    std::size_t count
) {
    std::vector<std::size_t> indices(values.size());

    for (std::size_t index = 0; index < indices.size(); ++index) {
        indices[index] = index;
    }

    std::partial_sort(
        indices.begin(),
        indices.begin() + static_cast<std::ptrdiff_t>(count),
        indices.end(),
        [&values](std::size_t left, std::size_t right) {
            return values[left] > values[right];
        }
    );

    indices.resize(count);
    return indices;
}

bool all_finite(const std::vector<float> & values, const char * name) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            std::cerr
                << "[FAIL] non-finite " << name
                << " at index " << index
                << '\n';
            return false;
        }
    }

    return true;
}

bool run_real_parity(
    const std::filesystem::path & model_path,
    const std::filesystem::path & fixture_path
) {
    parity_fixture fixture{};

    if (!read_fixture(fixture_path, &fixture)) {
        return false;
    }

    const terlml::hgrn_model_storage storage =
        terlml::hgrn_model_storage::load_f16_safetensors(model_path);

    terlml::hgrn_model_shape shape = storage.shape();
    shape.batch = 1;
    shape.sequence = 1;

    if (shape.vocab != fixture.logits.size()) {
        std::cerr
            << "[FAIL] checkpoint vocab=" << shape.vocab
            << ", fixture logits=" << fixture.logits.size()
            << '\n';
        return false;
    }

    const std::size_t conv_count =
        shape.layers *
        shape.batch *
        shape.hidden *
        shape.conv_kernel_size;
    const std::size_t recurrent_count =
        shape.layers * shape.batch * shape.hidden;

    if (conv_count != fixture.conv_state.size() ||
        recurrent_count != fixture.recurrent_state.size()) {
        std::cerr << "[FAIL] checkpoint state shape differs from fixture\n";
        return false;
    }

    std::vector<std::uint32_t> token_ids = {fixture.token_id};
    std::vector<float> initial_conv_state(conv_count, 0.0f);
    std::vector<float> initial_recurrent_state(recurrent_count, 0.0f);
    std::vector<float> actual_logits(shape.vocab, 0.0f);
    std::vector<float> actual_conv_state(conv_count, 0.0f);
    std::vector<float> actual_recurrent_state(recurrent_count, 0.0f);

    const char * trace_path = std::getenv("TERLML_CPP_LAYER_TRACE_PATH");
    std::ofstream trace_stream;
    layer_trace_writer trace_writer{};

    if (trace_path != nullptr) {
        trace_stream.open(trace_path, std::ios::binary);

        if (!trace_stream) {
            std::cerr << "[FAIL] open C++ layer trace: " << trace_path << '\n';
            return false;
        }

        constexpr std::array<char, 8> layer_magic = {
            'T', 'E', 'R', 'L', 'M', 'L', 'Y', 'R'
        };
        const std::uint32_t version = 1;
        const std::uint32_t layers = static_cast<std::uint32_t>(shape.layers);
        const std::uint32_t hidden = static_cast<std::uint32_t>(shape.hidden);
        const std::uint32_t kernel =
            static_cast<std::uint32_t>(shape.conv_kernel_size);

        trace_stream.write(layer_magic.data(), layer_magic.size());
        trace_stream.write(
            reinterpret_cast<const char *>(&version),
            static_cast<std::streamsize>(sizeof(version))
        );
        trace_stream.write(
            reinterpret_cast<const char *>(&layers),
            static_cast<std::streamsize>(sizeof(layers))
        );
        trace_stream.write(
            reinterpret_cast<const char *>(&hidden),
            static_cast<std::streamsize>(sizeof(hidden))
        );
        trace_stream.write(
            reinterpret_cast<const char *>(&kernel),
            static_cast<std::streamsize>(sizeof(kernel))
        );

        trace_writer.stream = &trace_stream;
    }

    terlml::hgrn_model_f32(
        token_ids.data(),
        storage.weights(),
        {
            .conv_cache = initial_conv_state.data(),
            .recurrent_state = initial_recurrent_state.data(),
        },
        actual_logits.data(),
        {
            .conv_cache = actual_conv_state.data(),
            .recurrent_state = actual_recurrent_state.data(),
        },
        1e-6f,
        shape,
        trace_path != nullptr ? &write_layer_trace : nullptr,
        trace_path != nullptr ? &trace_writer : nullptr
    );

    if (trace_path != nullptr) {
        trace_stream.close();

        if (!trace_stream) {
            std::cerr << "[FAIL] write C++ layer trace: " << trace_path << '\n';
            return false;
        }

        std::cout << "cpp_layer_trace=" << trace_path << '\n';
    }

    if (!all_finite(actual_logits, "logit") ||
        !all_finite(actual_conv_state, "convolution state") ||
        !all_finite(actual_recurrent_state, "recurrent state")) {
        return false;
    }

    print_error_summary("logits", actual_logits, fixture.logits);
    print_error_summary(
        "convolution_state",
        actual_conv_state,
        fixture.conv_state
    );
    print_error_summary(
        "recurrent_state",
        actual_recurrent_state,
        fixture.recurrent_state
    );

    const std::vector<std::size_t> actual_top_ids = top_indices(
        actual_logits,
        fixture.top_ids.size()
    );

    std::cout << "python_top10:";
    for (std::size_t index = 0; index < fixture.top_ids.size(); ++index) {
        std::cout
            << ' ' << fixture.top_ids[index]
            << '(' << fixture.top_logits[index] << ')';
    }
    std::cout << '\n';

    std::cout << "cpp_top10:";
    for (const std::size_t token_id : actual_top_ids) {
        std::cout
            << ' ' << token_id
            << '(' << actual_logits[token_id] << ')';
    }
    std::cout << '\n';

    const bool top1_matches = actual_top_ids.front() == fixture.top_ids.front();

    std::cout
        << "top1_match=" << (top1_matches ? "true" : "false")
        << " cpp_token=" << actual_top_ids.front()
        << " python_token=" << fixture.top_ids.front()
        << '\n';

    return true;
}

} // namespace

int main() {
    const char * model_path = std::getenv("TERLML_TEST_MODEL_PATH");
    const char * fixture_path = std::getenv("TERLML_PARITY_FIXTURE_PATH");

    if (model_path == nullptr || fixture_path == nullptr) {
        std::cout
            << "[SKIP] set TERLML_TEST_MODEL_PATH and "
            << "TERLML_PARITY_FIXTURE_PATH for real-weight parity diagnostics\n";
        return EXIT_SUCCESS;
    }

    if (!run_real_parity(model_path, fixture_path)) {
        std::cerr << "\nTerLM real-weight parity diagnostics failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nTerLM real-weight parity diagnostics completed.\n";
    return EXIT_SUCCESS;
}
