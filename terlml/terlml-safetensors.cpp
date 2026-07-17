#include "terlml-safetensors.h"

#include "json.hpp"

#include <bit>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace terlml {

namespace {

constexpr std::uint64_t k_max_header_bytes = 64U * 1024U * 1024U;
constexpr std::uint64_t k_f16_bytes = 2;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("SafeTensors: " + message);
}

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right,
    const std::string & context
) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        fail("integer overflow while " + context);
    }

    return left + right;
}

std::uint64_t checked_multiply(
    std::uint64_t left,
    std::uint64_t right,
    const std::string & context
) {
    if (left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        fail("integer overflow while " + context);
    }

    return left * right;
}

std::uint64_t read_u64_le(std::ifstream & file) {
    unsigned char bytes[8] = {};

    file.read(reinterpret_cast<char *>(bytes), sizeof(bytes));

    if (file.gcount() != static_cast<std::streamsize>(sizeof(bytes))) {
        fail("file is too small to contain a header length");
    }

    std::uint64_t value = 0;

    for (std::size_t index = 0; index < sizeof(bytes); ++index) {
        value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8);
    }

    return value;
}

float f16_to_f32(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16;
    std::uint32_t exponent = (bits >> 10) & 0x1fU;
    std::uint32_t mantissa = bits & 0x03ffU;
    std::uint32_t result = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            exponent = 1;

            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1;
                --exponent;
            }

            mantissa &= 0x03ffU;
            result = sign |
                ((exponent + 112U) << 23) |
                (mantissa << 13);
        }
    } else if (exponent == 0x1fU) {
        result = sign | 0x7f800000U | (mantissa << 13);
    } else {
        result = sign |
            ((exponent + 112U) << 23) |
            (mantissa << 13);
    }

    return std::bit_cast<float>(result);
}

} // namespace

safetensors_file::safetensors_file(const std::filesystem::path & path)
    : path_(path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path_, error);

    if (error) {
        fail("cannot read file size for '" + path_.string() + "'");
    }

    if (size > std::numeric_limits<std::uint64_t>::max()) {
        fail("file is too large");
    }

    file_size_ = static_cast<std::uint64_t>(size);

    std::ifstream file(path_, std::ios::binary);

    if (!file) {
        fail("cannot open '" + path_.string() + "'");
    }

    const std::uint64_t header_bytes = read_u64_le(file);

    if (header_bytes > k_max_header_bytes) {
        fail("header exceeds 64 MiB limit");
    }

    data_start_ = checked_add(8, header_bytes, "computing data start");

    if (data_start_ > file_size_) {
        fail("header extends beyond end of file");
    }

    std::string header(static_cast<std::size_t>(header_bytes), '\0');
    file.read(header.data(), static_cast<std::streamsize>(header.size()));

    if (file.gcount() != static_cast<std::streamsize>(header.size())) {
        fail("unable to read complete JSON header");
    }

    nlohmann::json root;

    try {
        root = nlohmann::json::parse(header);
    } catch (const nlohmann::json::exception & exception) {
        fail("invalid JSON header: " + std::string(exception.what()));
    }

    if (!root.is_object()) {
        fail("JSON header must be an object");
    }

    for (const auto & entry : root.items()) {
        const std::string & name = entry.key();
        const nlohmann::json & value = entry.value();

        if (name == "__metadata__") {
            continue;
        }

        if (!value.is_object()) {
            fail("tensor '" + name + "' metadata must be an object");
        }

        try {
            safetensors_tensor_info info = {
                .dtype = value.at("dtype").get<std::string>(),
                .shape = value.at("shape").get<std::vector<std::size_t>>(),
                .begin = value.at("data_offsets").at(0).get<std::uint64_t>(),
                .end = value.at("data_offsets").at(1).get<std::uint64_t>(),
            };

            if (info.end < info.begin) {
                fail("tensor '" + name + "' has reversed data offsets");
            }

            std::uint64_t element_count = 1;

            for (const std::size_t dimension : info.shape) {
                element_count = checked_multiply(
                    element_count,
                    dimension,
                    "counting elements in tensor '" + name + "'"
                );
            }

            if (info.dtype == "F16") {
                const std::uint64_t expected_bytes = checked_multiply(
                    element_count,
                    k_f16_bytes,
                    "computing byte count for tensor '" + name + "'"
                );

                if (info.end - info.begin != expected_bytes) {
                    fail(
                        "tensor '" + name +
                        "' data range does not match F16 shape"
                    );
                }
            }

            const std::uint64_t absolute_end = checked_add(
                data_start_,
                info.end,
                "computing data end for tensor '" + name + "'"
            );

            if (absolute_end > file_size_) {
                fail("tensor '" + name + "' extends beyond end of file");
            }

            if (!tensors_.emplace(name, std::move(info)).second) {
                fail("duplicate tensor name '" + name + "'");
            }
        } catch (const nlohmann::json::exception & exception) {
            fail(
                "invalid metadata for tensor '" + name + "': " +
                std::string(exception.what())
            );
        }
    }

    if (tensors_.empty()) {
        fail("header contains no tensors");
    }
}

const safetensors_tensor_info &
safetensors_file::tensor_info(const std::string & name) const {
    const auto iterator = tensors_.find(name);

    if (iterator == tensors_.end()) {
        fail("tensor not found: '" + name + "'");
    }

    return iterator->second;
}

std::vector<float>
safetensors_file::read_f16_as_f32(const std::string & name) const {
    const safetensors_tensor_info & info = tensor_info(name);

    if (info.dtype != "F16") {
        fail("tensor '" + name + "' is not F16");
    }

    const std::uint64_t byte_count = info.end - info.begin;

    if (byte_count % k_f16_bytes != 0) {
        fail("tensor '" + name + "' has an odd F16 byte count");
    }

    const std::uint64_t element_count = byte_count / k_f16_bytes;

    if (element_count > std::numeric_limits<std::size_t>::max()) {
        fail("tensor '" + name + "' is too large for this platform");
    }

    std::ifstream file(path_, std::ios::binary);

    if (!file) {
        fail("cannot reopen '" + path_.string() + "'");
    }

    const std::uint64_t absolute_begin = checked_add(
        data_start_,
        info.begin,
        "computing data offset for tensor '" + name + "'"
    );

    file.seekg(static_cast<std::streamoff>(absolute_begin), std::ios::beg);

    if (!file) {
        fail("cannot seek to tensor '" + name + "'");
    }

    std::vector<unsigned char> bytes(static_cast<std::size_t>(byte_count));
    file.read(
        reinterpret_cast<char *>(bytes.data()),
        static_cast<std::streamsize>(bytes.size())
    );

    if (file.gcount() != static_cast<std::streamsize>(bytes.size())) {
        fail("unable to read tensor '" + name + "'");
    }

    std::vector<float> values(static_cast<std::size_t>(element_count));

    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::uint16_t bits =
            static_cast<std::uint16_t>(bytes[index * 2]) |
            (static_cast<std::uint16_t>(bytes[index * 2 + 1]) << 8);

        values[index] = f16_to_f32(bits);
    }

    return values;
}

} // namespace terlml
