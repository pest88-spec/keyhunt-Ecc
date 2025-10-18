#pragma once

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

namespace test_util {

inline void SkipIfNoCudaDevice() {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess) {
    GTEST_SKIP() << "CUDA device query failed: " << cudaGetErrorString(status);
  }
  if (device_count <= 0) {
    GTEST_SKIP() << "No CUDA devices available";
  }
}

}  // namespace test_util
