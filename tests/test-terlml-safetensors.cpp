#include "../terlml/terlml-safetensors.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float k_tolerance = 1e-6f;

void write_u64_le(std::ofstream & file, std::uint64_t value) {
    for (std::size_t index = 0; index < 8; ++index) {
        file.put(static_cast<char>((value >> (index * 8)) & 0xffU));
    }
}

void write_u16_le(std::ofstream & file, std::uint16_t value) {
    file.put(static_cast<char>(value & 0xffU));
    file.put(static_cast<char>((value >> 8) & 0xffU));
}

bool expect(bool condition, const std::string & message) {
    if (!condition) {
        std::cerr << "[FAIL] " << message << '\n';
        return false;
    }

    std::cout << "[PASS] " << message << '\n';
    return true;
}

bool expect_close(
    const std::vector<float> & actual,
    const std::vector<float> & expected
) {
    if (actual.size() != expected.size()) {
        return expect(false, "FP16 output size");
    }

    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::fabs(actual[index] - expected[index]) > k_tolerance) {
            return expect(false, "FP16 value at index " + std::to_string(index));
        }
    }

    return expect(true, "FP16 conversion");
}

bool test_safetensors_reader() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "terlml-safetensors-test.safetensors";

    const std::string header =
        R"({"test.weight":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]}})";

    {
        std::ofstream file(path, std::ios::binary);

        if (!file) {
            return expect(false, "create temporary SafeTensors file");
        }

        write_u64_le(file, header.size());
        file.write(header.data(), static_cast<std::streamsize>(header.size()));

        write_u16_le(file, 0x3c00U);
        write_u16_le(file, 0xc000U);
        write_u16_le(file, 0x3800U);
        write_u16_le(file, 0x0000U);
    }

    try {
        const terlml::safetensors_file file(path);
        const auto & info = file.tensor_info("test.weight");
        const std::vector<float> values = file.read_f16_as_f32("test.weight");

        bool missing_tensor_rejected = false;

        try {
            static_cast<void>(file.tensor_info("does.not.exist"));
        } catch (const std::runtime_error &) {
            missing_tensor_rejected = true;
        }

        std::filesystem::remove(path);

        return expect(info.dtype == "F16", "tensor dtype") &&
            expect(info.shape == std::vector<std::size_t>({2, 2}), "tensor shape") &&
            expect(info.begin == 0 && info.end == 8, "tensor offsets") &&
            expect_close(values, {1.0f, -2.0f, 0.5f, 0.0f}) &&
            expect(missing_tensor_rejected, "missing tensor rejected");
    } catch (...) {
        std::filesystem::remove(path);
        throw;
    }
}

} // namespace

int main() {
    if (!test_safetensors_reader()) {
        std::cerr << "\nSafeTensors tests failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nAll SafeTensors tests passed.\n";
    return EXIT_SUCCESS;
}
