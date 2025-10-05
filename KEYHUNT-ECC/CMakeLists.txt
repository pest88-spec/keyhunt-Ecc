cmake_minimum_required(VERSION 3.18)
project(keyhunt_ecc LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

add_library(keyhunt_ecc STATIC
  core/batch_kernel.h
  core/ec_point.h
  core/fp_montgomery.h
  core/multiplicative_chain.h
  secp256k1/constants.h
  api/bridge.cu
)

target_include_directories(keyhunt_ecc PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compile options
if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  target_compile_options(keyhunt_ecc PRIVATE --expt-relaxed-constexpr)
endif()
