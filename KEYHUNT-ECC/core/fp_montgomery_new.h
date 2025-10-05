// Rewritten Montgomery multiplication based on exact gECC structure
// Temporary file for testing

#pragma once
#include <stdint.h>
#include "../secp256k1/constants.h"

#ifdef __GNUC__
#  define asm __asm__ __volatile__
#else
#  define asm asm volatile
#endif

namespace kh_ecc {

// Generic mul_n: multiplies a[0,2,4,6] or a[1,3,5,7] depending on pointer offset
__device__ __forceinline__ static void mul_n(uint32_t* acc, const uint32_t* a, uint32_t bi) {
  #pragma unroll
  for (int j = 0; j < 8; j += 2) {
    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
        : "=r"(acc[j]), "=r"(acc[j+1])
        : "r"(a[j]), "r"(bi));
  }
}

__device__ __forceinline__ static void madc_n_rshift(uint32_t* odd, const uint32_t* a, uint32_t bi) {
  #pragma unroll
  for (int j = 0; j < 6; j += 2) {
    asm("madc.lo.cc.u32 %0, %2, %3, %4; madc.hi.cc.u32 %1, %2, %3, %5;"
        : "=r"(odd[j]), "=r"(odd[j+1])
        : "r"(a[j]), "r"(bi), "r"(odd[j+2]), "r"(odd[j+3]));
  }
  asm("madc.lo.cc.u32 %0, %2, %3, %4; madc.hi.cc.u32 %1, %2, %3, 0;"
      : "=r"(odd[6]), "=r"(odd[7])
      : "r"(a[6]), "r"(bi), "r"(odd[8]));
}

__device__ __forceinline__ static void cmad_n(uint32_t* acc, const uint32_t* a, uint32_t bi) {
  asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
      : "+r"(acc[0]), "+r"(acc[1])
      : "r"(a[0]), "r"(bi));
  #pragma unroll
  for (int j = 2; j < 8; j += 2) {
    asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
        : "+r"(acc[j]), "+r"(acc[j+1])
        : "r"(a[j]), "r"(bi));
  }
}

// Direct port of gECC's mad_n_redc
__device__ __forceinline__ static void mad_n_redc(
    uint32_t* even, uint32_t* odd,
    const uint32_t* a,  // Full array a[0..7]
    uint32_t bi,        // Current scalar from b
    const uint32_t* p,  // Modulus
    bool first)
{
  if (first) {
    // even = a_even * bi, odd = a_odd * bi
    mul_n(odd, a+1, bi);  // odd = a[1,3,5,7] * bi
    mul_n(even, a, bi);   // even = a[0,2,4,6] * bi
  } else {
    // Shift: even[0] += odd[1], then recompute odd with shift
    asm("add.cc.u32 %0, %0, %1;" : "+r"(even[0]) : "r"(odd[1]));
    madc_n_rshift(odd, a+1, bi);  // a+1 means a[1,3,5,7]
    asm("addc.u32 %0, 0, 0;" : "+r"(odd[8]));
    // Accumulate into even
    cmad_n(even, a, bi);  // a means a[0,2,4,6]
    asm("addc.u32 %0, %0, 0;" : "+r"(even[8]));
  }

  // Montgomery reduction
  uint32_t m = even[0] * 0xD2253531u; // KH_PINV()

  // odd += p_odd * m, even += p_even * m
  // p_odd means p[1,3,5,7], p_even means p[0,2,4,6]
  // secp256k1 p = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
  // LE32: [FFFFFC2F, FFFFFFFE, FFFFFFFF, FFFFFFFF, FFFFFFFF, FFFFFFFF, FFFFFFFF, FFFFFFFF]
  const uint32_t p0 = 0xFFFFFC2Fu, p1 = 0xFFFFFFFEu, p2_7 = 0xFFFFFFFFu;

  asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
      : "+r"(odd[0]), "+r"(odd[1])
      : "r"(p1), "r"(m));
  #pragma unroll
  for (int j = 2; j < 8; j += 2) {
    asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
        : "+r"(odd[j]), "+r"(odd[j+1])
        : "r"(p2_7), "r"(m));
  }
  asm("addc.u32 %0, %0, 0;" : "+r"(odd[8]));

  asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
      : "+r"(even[0]), "+r"(even[1])
      : "r"(p0), "r"(m));
  #pragma unroll
  for (int j = 2; j < 8; j += 2) {
    asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
        : "+r"(even[j]), "+r"(even[j+1])
        : "r"(p2_7), "r"(m));
  }
  asm("addc.u32 %0, %0, 0;" : "+r"(even[8]));
}

__device__ __forceinline__ static void mont_mul_cios(uint32_t res[8], const uint32_t a[8], const uint32_t b[8]) {
  const uint32_t p0 = 0xFFFFFC2Fu, p1 = 0xFFFFFFFEu, p2_7 = 0xFFFFFFFFu;

  uint32_t even[9], odd[9];
  even[8] = 0;
  odd[8] = 0;

  #pragma unroll
  for (int i = 0; i < 8; i += 2) {
    // First call: even/odd, b[i]
    mad_n_redc(even, odd, a, b[i], nullptr, i==0);
    // Second call: SWAP - odd/even, b[i+1]
    mad_n_redc(odd, even, a, b[i+1], nullptr, false);
  }

  // Merge even and odd
  asm("add.cc.u32 %0, %1, %2;" : "=r"(res[0]) : "r"(even[0]), "r"(odd[1]));
  #pragma unroll
  for (int i = 1; i < 8; i++) {
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(res[i]) : "r"(even[i]), "r"(odd[i+1]));
  }
  asm("addc.cc.u32 %0, %0, 0;" : "+r"(even[8]));

  // Final subtraction
  uint32_t sub[8];
  asm("sub.cc.u32 %0, %1, %2;" : "=r"(sub[0]) : "r"(res[0]), "r"(p0));
  asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub[1]) : "r"(res[1]), "r"(p1));
  #pragma unroll
  for (int i = 2; i < 8; i++) {
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub[i]) : "r"(res[i]), "r"(p2_7));
  }
  asm("subc.u32 %0, %0, 0;" : "+r"(even[8]));

  // Conditional move
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (even[8] == 0) res[i] = sub[i];
  }
}

} // namespace kh_ecc
