#pragma once
#include <stdint.h>
#include "device_functions.h"

namespace kh_ecc {
namespace cuda_layout {

// Simplified memory layout: 1 lane per slot (WIDTH = 1). This avoids subwarp logic for now.
struct Layout1 {
  static constexpr uint32_t WIDTH = 1;
  static constexpr uint32_t LOG_WIDTH = 0;

  __device__ __forceinline__ static uint32_t lane_idx() { return 0u; }
  __device__ __forceinline__ static uint32_t slot_idx() { return 0u; }
  __device__ __forceinline__ static uint32_t global_slot_idx() { return 0u; }

  __device__ __forceinline__ static uint32_t ballot(bool b) { return b ? 1u : 0u; }
};

} // namespace cuda_layout
} // namespace kh_ecc
