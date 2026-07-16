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
