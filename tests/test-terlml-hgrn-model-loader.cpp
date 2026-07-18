#include "../terlml/terlml-hgrn-model-loader.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void write_u64_le(std::ofstream & file, std::uint64_t value) {
    for (std::size_t index = 0; index < 8; ++index) {
        file.put(static_cast<char>((value >> (index * 8)) & 0xffU));
    }
}

bool test_rejects_wrong_embedding_shape() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "terlml-hgrn-model-loader-wrong-shape.safetensors";

    const std::string header =
        R"({"model.embeddings.weight":{"dtype":"F16","shape":[1,1],"data_offsets":[0,2]}})";

    {
        std::ofstream file(path, std::ios::binary);

        if (!file) {
            std::cerr << "[FAIL] create test file\n";
            return false;
        }

        write_u64_le(file, header.size());
        file.write(header.data(), static_cast<std::streamsize>(header.size()));
        file.put('\0');
        file.put('\0');
    }

    bool rejected = false;

    try {
        static_cast<void>(
            terlml::hgrn_model_storage::load_f16_safetensors(path)
        );
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find(
            "model.embeddings.weight"
        ) != std::string::npos;
    }

    std::filesystem::remove(path);

    if (!rejected) {
        std::cerr << "[FAIL] wrong embedding shape rejected\n";
        return false;
    }

    std::cout << "[PASS] wrong embedding shape rejected\n";
    return true;
}

} // namespace

int main() {
    if (!test_rejects_wrong_embedding_shape()) {
        return EXIT_FAILURE;
    }

    std::cout << "\nAll HGRN model-loader tests passed.\n";
    return EXIT_SUCCESS;
}
