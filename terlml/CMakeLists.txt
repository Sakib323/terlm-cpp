add_library(terlml-hgrn-reference STATIC
    terlml-hgrn-reference.cpp
)

target_include_directories(terlml-hgrn-reference
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(terlml-hgrn-reference PUBLIC cxx_std_20)

add_executable(test-terlml-hgrn
    ${CMAKE_SOURCE_DIR}/tests/test-terlml-hgrn.cpp
)

target_link_libraries(test-terlml-hgrn
    PRIVATE
        terlml-hgrn-reference
)

add_test(
    NAME terlml-hgrn-reference
    COMMAND test-terlml-hgrn
)

add_executable(test-terlml-hgrn-parity
    ${CMAKE_SOURCE_DIR}/tests/test-terlml-hgrn-parity.cpp
)

target_link_libraries(test-terlml-hgrn-parity
    PRIVATE
        terlml-hgrn-reference
)

add_test(
    NAME terlml-hgrn-pytorch-parity
    COMMAND test-terlml-hgrn-parity
            ${CMAKE_SOURCE_DIR}/tests/fixtures/hgrn-parity-f32.bin
)

add_library(terlml-hgrn-neon STATIC
    terlml-hgrn-neon.cpp
)

target_include_directories(terlml-hgrn-neon
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(terlml-hgrn-neon PUBLIC cxx_std_20)

add_executable(test-terlml-hgrn-neon
    ${CMAKE_SOURCE_DIR}/tests/test-terlml-hgrn-neon.cpp
)

target_link_libraries(test-terlml-hgrn-neon
    PRIVATE
        terlml-hgrn-reference
        terlml-hgrn-neon
)

add_test(
    NAME terlml-hgrn-neon-parity
    COMMAND test-terlml-hgrn-neon
)

add_library(terlml-causal-conv STATIC
    terlml-causal-conv.cpp
)

target_include_directories(terlml-causal-conv
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(terlml-causal-conv PUBLIC cxx_std_20)

add_executable(test-terlml-causal-conv
    ${CMAKE_SOURCE_DIR}/tests/test-terlml-causal-conv.cpp
)

target_link_libraries(test-terlml-causal-conv
    PRIVATE
        terlml-causal-conv
)

add_test(
    NAME terlml-causal-conv
    COMMAND test-terlml-causal-conv
)

add_library(terlml-rmsnorm-gate STATIC
    terlml-rmsnorm-gate.cpp
)

target_include_directories(terlml-rmsnorm-gate
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(terlml-rmsnorm-gate PUBLIC cxx_std_20)

add_executable(test-terlml-rmsnorm-gate
    ${CMAKE_SOURCE_DIR}/tests/test-terlml-rmsnorm-gate.cpp
)

target_link_libraries(test-terlml-rmsnorm-gate
    PRIVATE
        terlml-rmsnorm-gate
)

add_test(
    NAME terlml-rmsnorm-gate
    COMMAND test-terlml-rmsnorm-gate
)

add_library(terlml-hgrn-attention-block STATIC
    terlml-hgrn-attention-block.cpp
)

target_include_directories(terlml-hgrn-attention-block
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(terlml-hgrn-attention-block PUBLIC cxx_std_20)

target_link_libraries(terlml-hgrn-attention-block
    PUBLIC
        terlml-causal-conv
        terlml-fused-bitlinear
        terlml-hgrn-reference
        terlml-rmsnorm-gate
)

add_executable(test-terlml-hgrn-attention-block
    ${CMAKE_SOURCE_DIR}/tests/test-terlml-hgrn-attention-block.cpp
)

target_link_libraries(test-terlml-hgrn-attention-block
    PRIVATE
        terlml-hgrn-attention-block
)

add_test(
    NAME terlml-hgrn-attention-block
    COMMAND test-terlml-hgrn-attention-block
)

add_executable(test-terlml-hgrn-attention-block-parity
    ${CMAKE_SOURCE_DIR}/tests/test-terlml-hgrn-attention-block-parity.cpp
)

target_link_libraries(test-terlml-hgrn-attention-block-parity
    PRIVATE
        terlml-hgrn-attention-block
)

add_test(
    NAME terlml-hgrn-attention-block-pytorch-parity
    COMMAND test-terlml-hgrn-attention-block-parity
            ${CMAKE_SOURCE_DIR}/tests/fixtures/hgrn-attention-block-parity-f32.bin
)

add_library(terlml-fused-bitlinear STATIC
    terlml-fused-bitlinear.cpp
)

target_include_directories(terlml-fused-bitlinear
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(terlml-fused-bitlinear PUBLIC cxx_std_20)

add_executable(test-terlml-fused-bitlinear-parity
    ${CMAKE_SOURCE_DIR}/tests/test-terlml-fused-bitlinear-parity.cpp
)

target_link_libraries(test-terlml-fused-bitlinear-parity
    PRIVATE
        terlml-fused-bitlinear
)

add_test(
    NAME terlml-fused-bitlinear-pytorch-parity
    COMMAND test-terlml-fused-bitlinear-parity
            ${CMAKE_SOURCE_DIR}/tests/fixtures/fused-bitlinear-parity-f32.bin
)

add_library(terlml-hgrn-mlp STATIC
    terlml-hgrn-mlp.cpp
)

target_include_directories(terlml-hgrn-mlp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(terlml-hgrn-mlp PUBLIC cxx_std_20)

target_link_libraries(terlml-hgrn-mlp
    PUBLIC
        terlml-fused-bitlinear
)

add_executable(test-terlml-hgrn-mlp-parity
    ${CMAKE_SOURCE_DIR}/tests/test-terlml-hgrn-mlp-parity.cpp
)

target_link_libraries(test-terlml-hgrn-mlp-parity
    PRIVATE
        terlml-hgrn-mlp
)

add_test(
    NAME terlml-hgrn-mlp-pytorch-parity
    COMMAND test-terlml-hgrn-mlp-parity
            ${CMAKE_SOURCE_DIR}/tests/fixtures/hgrn-mlp-parity-f32.bin
)

add_library(terlml-hgrn-decoder-layer STATIC
    terlml-hgrn-decoder-layer.cpp
)

target_include_directories(terlml-hgrn-decoder-layer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(terlml-hgrn-decoder-layer PUBLIC cxx_std_20)

target_link_libraries(terlml-hgrn-decoder-layer
    PUBLIC
        terlml-hgrn-attention-block
        terlml-hgrn-mlp
)

add_executable(test-terlml-hgrn-decoder-layer
    ${CMAKE_SOURCE_DIR}/tests/test-terlml-hgrn-decoder-layer.cpp
)

target_link_libraries(test-terlml-hgrn-decoder-layer
    PRIVATE
        terlml-hgrn-decoder-layer
)

add_test(
    NAME terlml-hgrn-decoder-layer
    COMMAND test-terlml-hgrn-decoder-layer
)

add_library(terlml-hgrn-lower-bounds STATIC
    terlml-hgrn-lower-bounds.cpp
)

target_include_directories(terlml-hgrn-lower-bounds
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(terlml-hgrn-lower-bounds PUBLIC cxx_std_20)

add_executable(test-terlml-hgrn-lower-bounds
    ${CMAKE_SOURCE_DIR}/tests/test-terlml-hgrn-lower-bounds.cpp
)

target_link_libraries(test-terlml-hgrn-lower-bounds
    PRIVATE
        terlml-hgrn-lower-bounds
)

add_test(
    NAME terlml-hgrn-lower-bounds
    COMMAND test-terlml-hgrn-lower-bounds
)

add_library(terlml-rmsnorm STATIC
    terlml-rmsnorm.cpp
)

target_include_directories(terlml-rmsnorm
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(terlml-rmsnorm PUBLIC cxx_std_20)

add_executable(test-terlml-rmsnorm
    ${CMAKE_SOURCE_DIR}/tests/test-terlml-rmsnorm.cpp
)

target_link_libraries(test-terlml-rmsnorm
    PRIVATE
        terlml-rmsnorm
)

add_test(
    NAME terlml-rmsnorm
    COMMAND test-terlml-rmsnorm
)

add_library(terlml-hgrn-model STATIC
    terlml-hgrn-model.cpp
)

target_include_directories(terlml-hgrn-model
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(terlml-hgrn-model PUBLIC cxx_std_20)

target_link_libraries(terlml-hgrn-model
    PUBLIC
        terlml-fused-bitlinear
        terlml-hgrn-decoder-layer
        terlml-hgrn-lower-bounds
        terlml-rmsnorm
)

add_executable(test-terlml-hgrn-model
    ${CMAKE_SOURCE_DIR}/tests/test-terlml-hgrn-model.cpp
)

target_link_libraries(test-terlml-hgrn-model
    PRIVATE
        terlml-hgrn-model
)

add_test(
    NAME terlml-hgrn-model
    COMMAND test-terlml-hgrn-model
)

add_executable(test-terlml-hgrn-model-parity
    ${CMAKE_SOURCE_DIR}/tests/test-terlml-hgrn-model-parity.cpp
)

target_link_libraries(test-terlml-hgrn-model-parity
    PRIVATE
        terlml-hgrn-model
)

add_test(
    NAME terlml-hgrn-model-pytorch-parity
    COMMAND test-terlml-hgrn-model-parity
            ${CMAKE_SOURCE_DIR}/tests/fixtures/hgrn-model-parity-f32.bin
)

add_library(terlml-safetensors STATIC
    terlml-safetensors.cpp
)

target_include_directories(terlml-safetensors
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/3rdparty/llama.cpp/common
)

target_compile_features(terlml-safetensors PUBLIC cxx_std_20)

add_executable(test-terlml-safetensors
    ${CMAKE_SOURCE_DIR}/tests/test-terlml-safetensors.cpp
)

target_link_libraries(test-terlml-safetensors
    PRIVATE
        terlml-safetensors
)

add_test(
    NAME terlml-safetensors
    COMMAND test-terlml-safetensors
)
