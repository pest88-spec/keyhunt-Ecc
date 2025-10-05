#pragma once

// Minimal CUDA device qualifier fallbacks to allow host-only parsing.
#ifndef __CUDACC__
#define __device__
#define __host__
#define __global__
#define __forceinline__
#endif

namespace kh_ecc {
namespace cuda_helpers {
// Place for small device utilities if needed later.
}
}