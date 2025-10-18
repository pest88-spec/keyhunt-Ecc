cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(keyhunt_ecc_superbuild LANGUAGES CUDA CXX)

include(CTest)

if(NOT DEFINED KEYHUNT_ECC_ENABLE_TESTS)
  set(KEYHUNT_ECC_ENABLE_TESTS ${BUILD_TESTING})
endif()

add_subdirectory(KEYHUNT-ECC)
