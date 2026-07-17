#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace terlml {

struct safetensors_tensor_info {
    std::string dtype;
    std::vector<std::size_t> shape;
    std::uint64_t begin;
    std::uint64_t end;
};

class safetensors_file {
public:
    explicit safetensors_file(const std::filesystem::path & path);

    const safetensors_tensor_info &
    tensor_info(const std::string & name) const;

    std::vector<float>
    read_f16_as_f32(const std::string & name) const;

private:
    std::filesystem::path path_;
    std::uint64_t data_start_ = 0;
    std::uint64_t file_size_ = 0;
    std::unordered_map<std::string, safetensors_tensor_info> tensors_;
};

} // namespace terlml
