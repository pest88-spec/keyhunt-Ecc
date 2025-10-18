cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(keyhunt_ecc LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Auto-detect CUDA architecture if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Option to enable/disable tests
option(KEYHUNT_ECC_ENABLE_TESTS "Build KEYHUNT-ECC tests" OFF)

# Locate Python3 for constant generation
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Locate CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Generate constants using scripts from parent directory
file(GLOB CONSTANTS_GENERATOR_SCRIPTS
  "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/constants_generator.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/constants.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/ccgen.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/ec_ops.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/ec.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/field.py"
)
set(GENERATED_CONSTANTS_HEADERS
  "${CMAKE_CURRENT_BINARY_DIR}/include/fp_constants.h"
  "${CMAKE_CURRENT_BINARY_DIR}/include/fp_ops_cc_details.h"
  "${CMAKE_CURRENT_BINARY_DIR}/include/ec_ops_add_details.h"
  "${CMAKE_CURRENT_BINARY_DIR}/include/ec_ops_dbl_details.h"
  "${CMAKE_CURRENT_BINARY_DIR}/include/fp_test_constants.h"
)
add_custom_command(OUTPUT ${GENERATED_CONSTANTS_HEADERS}
  COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/constants_generator.py" 
          --out "${CMAKE_CURRENT_BINARY_DIR}/include"
  DEPENDS ${CONSTANTS_GENERATOR_SCRIPTS}
  COMMENT "Generating KEYHUNT-ECC constants..."
  VERBATIM
)
add_custom_target(keyhunt_ecc_generated_constants DEPENDS ${GENERATED_CONSTANTS_HEADERS})

# Export generated include directory for tests and other consumers
set(KEYHUNT_ECC_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include" CACHE INTERNAL "")

# Main library target
add_library(keyhunt_ecc STATIC
  core/batch_kernel.h
  core/ec_point.h
  core/fp_montgomery.h
  core/multiplicative_chain.h
  secp256k1/constants.h
  api/bridge.cu
)

add_dependencies(keyhunt_ecc keyhunt_ecc_generated_constants)

target_include_directories(keyhunt_ecc PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_BINARY_DIR}/include
)

target_link_libraries(keyhunt_ecc PUBLIC CUDA::cudart)

# Compile options
if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  target_compile_options(keyhunt_ecc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

# Optionally include tests (controlled by parent or standalone builds)
if(KEYHUNT_ECC_ENABLE_TESTS)
  include(GoogleTest)
  enable_testing()
  find_package(GTest REQUIRED)
  add_subdirectory(../test ${CMAKE_CURRENT_BINARY_DIR}/test)
endif()
