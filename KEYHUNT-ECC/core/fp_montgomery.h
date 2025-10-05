#pragma once
#include <stdint.h>
#include <stddef.h>
#include "../cuda/device_functions.h"
#include "../secp256k1/constants.h"

namespace kh_ecc {
namespace secp256k1 {

// 为 CUDA 设备编译路径提供常量内联访问器，避免对类内 constexpr 数组的 ODR 要求
#if defined(__CUDA_ARCH__)
__device__ __forceinline__ uint32_t KH_P_at(int i) {
  switch(i){
    case 0:return 0xFFFFFC2Fu; case 1:return 0xFFFFFFFEu; case 2:return 0xFFFFFFFFu; case 3:return 0xFFFFFFFFu;
    case 4:return 0xFFFFFFFFu; case 5:return 0xFFFFFFFFu; case 6:return 0xFFFFFFFFu; default:return 0xFFFFFFFFu;
  }
}
__device__ __forceinline__ uint32_t KH_R_at(int i) {
  switch(i){
    case 0:return 0x000003D1u; case 1:return 0x00000001u; case 2:return 0u; case 3:return 0u;
    case 4:return 0u; case 5:return 0u; case 6:return 0u; default:return 0u;
  }
}
__device__ __forceinline__ uint32_t KH_R2_at(int i) {
  switch(i){
    case 0:return 0x000E90A1u; case 1:return 0x000007A2u; case 2:return 0x00000001u; case 3:return 0u;
    case 4:return 0u; case 5:return 0u; case 6:return 0u; default:return 0u;
  }
}
__device__ __forceinline__ uint32_t KH_ONE_at(int i) {
  return (i==0)?1u:0u;
}
__device__ __forceinline__ uint32_t KH_PINV() { return 0xD2253531u; }
#endif

// Field element in Fp for secp256k1, 256-bit stored as 8x32-bit limbs (LE).
struct Fp {
  static constexpr size_t LIMBS = 8;
  static constexpr size_t REXP  = 8; // digits per element for 32-bit base
  uint32_t d[LIMBS];

  // Constructors
  __device__ __forceinline__ static Fp zero() {
    Fp r{}; // zero-initialized
    return r;
  }
  __device__ __forceinline__ static Fp one() {
    Fp r{}; r.d[0] = 1u; return r;
  }

  // Basic helpers (Montgomery constants)
  __device__ __forceinline__ static Fp mont_one() {
    Fp r{};
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
#if defined(__CUDA_ARCH__)
      r.d[i] = KH_R_at(i);
#else
      r.d[i] = Params::R[i];
#endif
    }
    return r;
  }

  // ===== PTX helpers (性能关键，__CUDA_ARCH__ 下使用 asm，CPU 下使用便携回退) =====
  __device__ __forceinline__ static void ptx_mul_n(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = REXP) {
  #if defined(__CUDA_ARCH__)
    // acc[0..n-1] = a[0..n-1] * bi (lo/hi 展开)，布局仿 gECC mul_n
    for (size_t j = 0; j < n; j += 2) {
      asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
          : "=r"(acc[j]), "=r"(acc[j+1])
          : "r"(a[j]), "r"(bi));
    }
  #else
    // 便携回退：逐元素 64 位乘法
    for (size_t j = 0; j < n; j += 2) {
      uint64_t p0 = (uint64_t)a[j] * bi;
      uint64_t p1 = (uint64_t)a[j+1] * bi;
      acc[j]   = (uint32_t)(p0 & 0xFFFFFFFFu);
      acc[j+1] = (uint32_t)((p0 >> 32) & 0xFFFFFFFFu);
      // 注意：这里未把 p1 与进位合并，仅作为占位；完整实现会用 cmad/madc 级联
      (void)p1;
    }
  #endif
  }

  __device__ __forceinline__ static void ptx_madc_n(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = REXP) {
  #if defined(__CUDA_ARCH__)
    // acc += a * bi，使用 mad*.u32 链式带进位累加
    asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
        : "+r"(acc[0]), "+r"(acc[1])
        : "r"(a[0]), "r"(bi));
    for (size_t j = 2; j < n; j += 2) {
      asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
          : "+r"(acc[j]), "+r"(acc[j+1])
          : "r"(a[j]), "r"(bi));
    }
  #else
    // 便携回退：64 位乘法 + 本地进位（占位）
    uint64_t carry = 0;
    for (size_t j = 0; j < n; ++j) {
      uint64_t sum = (uint64_t)acc[j] + (uint64_t)a[j] * bi + carry;
      acc[j] = (uint32_t)(sum & 0xFFFFFFFFu);
      carry  = (sum >> 32);
    }
  #endif
  }

  // 带出进位的乘加：acc += a * bi，返回末位进位到 carry_out
  __device__ __forceinline__ static void ptx_madc_n_carry(uint32_t* acc, const uint32_t* a, uint32_t bi, uint32_t &carry_out, size_t n = REXP) {
  #if defined(__CUDA_ARCH__)
    asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
        : "+r"(acc[0]), "+r"(acc[1])
        : "r"(a[0]), "r"(bi));
    for (size_t j = 2; j < n; j += 2) {
      asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
          : "+r"(acc[j]), "+r"(acc[j+1])
          : "r"(a[j]), "r"(bi));
    }
    // 提取最终进位到寄存器
    asm("addc.u32 %0, 0, 0;" : "=r"(carry_out));
  #else
    uint64_t carry = 0;
    for (size_t j = 0; j < n; ++j) {
      uint64_t sum = (uint64_t)acc[j] + (uint64_t)a[j] * bi + carry;
      acc[j] = (uint32_t)sum;
      carry = sum >> 32;
    }
    carry_out = (uint32_t)carry;
  #endif
  }

  // Compare a >= b (unsigned, little-endian)
  __device__ __forceinline__ static bool geq(const uint32_t a[8], const uint32_t b[8]) {
    for (int i = 7; i >= 0; --i) {
      if (a[i] != b[i]) return a[i] > b[i];
    }
    return true; // equal
  }
#if defined(__CUDA_ARCH__)
  // Compare a >= P (device side, inline fetch of P)
  __device__ __forceinline__ static bool geq_p(const uint32_t a[8]) {
    for (int i = 7; i >= 0; --i) {
      uint32_t pi = KH_P_at(i);
      if (a[i] != pi) return a[i] > pi;
    }
    return true;
  }
#endif

  // c = a - b (mod p)
  __device__ __forceinline__ static void sub_n(uint32_t c[8], const uint32_t a[8], const uint32_t b[8]) {
    uint64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint64_t ai = a[i];
      uint64_t bi = b[i];
      uint64_t t = ai - bi - borrow;
      c[i] = (uint32_t)t;
      borrow = (t >> 63) & 1ull; // if underflow, high bit set
    }
    if (borrow) {
      // add back p
      uint64_t carry = 0;
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        uint64_t t = (uint64_t)c[i] + (uint64_t)(
#if defined(__CUDA_ARCH__)
          KH_P_at(i)
#else
          Params::P[i]
#endif
        ) + carry;
        c[i] = (uint32_t)t;
        carry = t >> 32;
      }
    }
  }

  // c = a + b (mod p)
  __device__ __forceinline__ static void add_n(uint32_t c[8], const uint32_t a[8], const uint32_t b[8]) {
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint64_t t = (uint64_t)a[i] + (uint64_t)b[i] + carry;
      c[i] = (uint32_t)t;
      carry = t >> 32;
    }
    // if overflow or c >= p then subtract p
#if defined(__CUDA_ARCH__)
    if (carry || geq_p(c)) {
#else
    if (carry || geq(c, Params::P)) {
#endif
      uint64_t borrow = 0;
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        uint64_t t = (uint64_t)c[i] - (uint64_t)(
#if defined(__CUDA_ARCH__)
          KH_P_at(i)
#else
          Params::P[i]
#endif
        ) - borrow;
        c[i] = (uint32_t)t;
        borrow = (t >> 63) & 1ull;
      }
    }
  }

  // C 版本的 CIOS Montgomery 乘法（便携回退）
  __device__ __forceinline__ static void mont_mul_c(uint32_t res[8], const uint32_t a[8], const uint32_t b[8]) {
    uint32_t n0 =
#if defined(__CUDA_ARCH__)
      KH_PINV()
#else
      Params::PINV
#endif
    ;

    // 修复: t[8] 需要 64 位以避免溢出截断
    uint32_t t[8];
    uint64_t t8 = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) t[i] = 0u;

    for (int i = 0; i < 8; ++i) {
      // t = t + a * b[i]
      uint64_t carry = 0;
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        uint64_t prod = (uint64_t)a[j] * (uint64_t)b[i];
        uint64_t sum = (uint64_t)t[j] + prod + carry;
        t[j] = (uint32_t)sum;
        carry = sum >> 32;
      }
      t8 += carry;  // 保留完整 64 位进位

      // m = (t[0] * n0) mod 2^32
      uint32_t m = (uint32_t)((uint64_t)t[0] * (uint64_t)n0);

      // t = t + m * p
      uint64_t carry2 = 0;
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        uint64_t prod = (uint64_t)m * (uint64_t)(
#if defined(__CUDA_ARCH__)
          KH_P_at(j)
#else
          Params::P[j]
#endif
        );
        uint64_t sum = (uint64_t)t[j] + prod + carry2;
        t[j] = (uint32_t)sum;
        carry2 = sum >> 32;
      }
      t8 += carry2;  // 保留完整 64 位进位

      // t = t >> 32 (drop least limb)
      #pragma unroll
      for (int j = 0; j < 7; ++j) t[j] = t[j+1];
      t[7] = (uint32_t)t8;  // 取 t8 的低 32 位
      t8 >>= 32;            // t8 保留高位
    }

    // conditional subtraction if t >= p
    // 如果 t8 > 0，说明 t 是 257+ 位，必然 >= P，需要减 P
    // 否则检查 t[0..7] >= P
#if defined(__CUDA_ARCH__)
    if (t8 > 0 || geq_p(t)) {
      // t - P（普通减法，不是模减法）
      uint64_t borrow = 0;
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        uint32_t pi = KH_P_at(i);
        uint64_t diff = (uint64_t)t[i] - (uint64_t)pi - borrow;
        res[i] = (uint32_t)diff;
        borrow = (diff >> 63) & 1ull;
      }
      // 注意：如果 t8 > 0，减 P 后 borrow 会被 t8 抵消，结果仍然 >= 0
    } else {
      #pragma unroll
      for (int j = 0; j < 8; ++j) res[j] = t[j];
    }
#else
    if (t8 > 0 || geq(t, Params::P)) {
      uint64_t borrow = 0;
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        uint64_t diff = (uint64_t)t[i] - (uint64_t)Params::P[i] - borrow;
        res[i] = (uint32_t)diff;
        borrow = (diff >> 63) & 1ull;
      }
    } else {
      #pragma unroll
      for (int j = 0; j < 8; ++j) res[j] = t[j];
    }
#endif
  }

  // PTX 优化版本的 CIOS Montgomery 乘法（使用 madc 链）
  // 参考 gECC CIOS 算法实现的 Montgomery 乘法
  // 来源溯源 (Provenance):
  // - 原始项目: gECC (https://arxiv.org/abs/2501.03245)
  // - 原始文件: include/gecc/arith/details/fp_mont_multiply.h
  // - 许可证: MIT
  // - 修改说明: 适配为静态函数风格,使用 8×uint32 表示
  __device__ __forceinline__ static void mont_mul_ptx(uint32_t res[8], const uint32_t a[8], const uint32_t b[8]) {
  #if defined(__CUDA_ARCH__)
    // CIOS (Coarsely Integrated Operand Scanning) Montgomery 乘法
    uint32_t even[9], odd[9];
    even[8] = 0;
    odd[8] = 0;

    #pragma unroll
    for (int i = 0; i < 8; i += 2) {
      // 处理 b[i] (even round)
      if (i == 0) {
        // 初始化: even = a * b[0] (分奇偶位)
        #pragma unroll
        for (int j = 0; j < 8; j += 2) {
          asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
              : "=r"(even[j]), "=r"(even[j+1])
              : "r"(a[j]), "r"(b[0]));
        }
        // odd = a[1,3,5,7] * b[0]
        #pragma unroll
        for (int j = 0; j < 8; j += 2) {
          asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
              : "=r"(odd[j]), "=r"(odd[j+1])
              : "r"(a[j+1]), "r"(b[0]));
        }
      } else {
        // even += a * b[i], 带进位链
        asm("add.cc.u32 %0, %0, %1;" : "+r"(even[0]) : "r"(odd[1]));
        #pragma unroll
        for (int j = 0; j < 6; j += 2) {
          asm("madc.lo.cc.u32 %0, %2, %3, %4; madc.hi.cc.u32 %1, %2, %3, %5;"
              : "=r"(odd[j]), "=r"(odd[j+1])
              : "r"(a[j+1]), "r"(b[i]), "r"(odd[j+2]), "r"(odd[j+3]));
        }
        asm("madc.lo.cc.u32 %0, %2, %3, %4; madc.hi.cc.u32 %1, %2, %3, 0;"
            : "=r"(odd[6]), "=r"(odd[7])
            : "r"(a[7]), "r"(b[i]), "r"(odd[8]));
        asm("addc.u32 %0, 0, 0;" : "=r"(odd[8]));

        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
            : "+r"(even[0]), "+r"(even[1])
            : "r"(a[0]), "r"(b[i]));
        #pragma unroll
        for (int j = 2; j < 8; j += 2) {
          asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
              : "+r"(even[j]), "+r"(even[j+1])
              : "r"(a[j]), "r"(b[i]));
        }
        asm("addc.u32 %0, %0, 0;" : "+r"(even[8]));
      }

      // Montgomery reduction: m = even[0] * pinv
      uint32_t m = even[0] * KH_PINV();

      // odd += p * m
      asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
          : "+r"(odd[0]), "+r"(odd[1])
          : "r"(KH_P_at(1)), "r"(m));
      #pragma unroll
      for (int j = 2; j < 8; j += 2) {
        asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
            : "+r"(odd[j]), "+r"(odd[j+1])
            : "r"(KH_P_at(j+1)), "r"(m));
      }
      asm("addc.u32 %0, %0, 0;" : "+r"(odd[8]));

      // even += p[0,2,4,6] * m
      asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
          : "+r"(even[0]), "+r"(even[1])
          : "r"(KH_P_at(0)), "r"(m));
      #pragma unroll
      for (int j = 2; j < 8; j += 2) {
        asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
            : "+r"(even[j]), "+r"(even[j+1])
            : "r"(KH_P_at(j)), "r"(m));
      }
      asm("addc.u32 %0, %0, 0;" : "+r"(even[8]));

      // 处理 b[i+1] (odd round) - 类似逻辑
      asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
          : "+r"(odd[0]), "+r"(odd[1])
          : "r"(a[1]), "r"(b[i+1]));
      #pragma unroll
      for (int j = 2; j < 8; j += 2) {
        asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
            : "+r"(odd[j]), "+r"(odd[j+1])
            : "r"(a[j+1]), "r"(b[i+1]));
      }
      asm("addc.u32 %0, %0, 0;" : "+r"(odd[8]));

      asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
          : "+r"(even[0]), "+r"(even[1])
          : "r"(a[0]), "r"(b[i+1]));
      #pragma unroll
      for (int j = 2; j < 8; j += 2) {
        asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
            : "+r"(even[j]), "+r"(even[j+1])
            : "r"(a[j]), "r"(b[i+1]));
      }
      asm("addc.u32 %0, %0, 0;" : "+r"(even[8]));

      // Montgomery reduction for b[i+1]
      m = odd[0] * KH_PINV();

      asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
          : "+r"(even[0]), "+r"(even[1])
          : "r"(KH_P_at(1)), "r"(m));
      #pragma unroll
      for (int j = 2; j < 8; j += 2) {
        asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
            : "+r"(even[j]), "+r"(even[j+1])
            : "r"(KH_P_at(j+1)), "r"(m));
      }
      asm("addc.u32 %0, %0, 0;" : "+r"(even[8]));

      asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
          : "+r"(odd[0]), "+r"(odd[1])
          : "r"(KH_P_at(0)), "r"(m));
      #pragma unroll
      for (int j = 2; j < 8; j += 2) {
        asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
            : "+r"(odd[j]), "+r"(odd[j+1])
            : "r"(KH_P_at(j)), "r"(m));
      }
      asm("addc.u32 %0, %0, 0;" : "+r"(odd[8]));
    }

    // 合并 even 和 odd
    uint32_t tmp[8];
    asm("add.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(even[0]), "r"(odd[1]));
    #pragma unroll
    for (int i = 1; i < 8; i++) {
      asm("addc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(even[i]), "r"(odd[i+1]));
    }
    asm("addc.cc.u32 %0, %0, 0;" : "+r"(even[8]));

    // 最终约简: if tmp >= p then tmp -= p
    uint32_t sub_res[8];
    uint32_t borrow;
    asm("sub.cc.u32 %0, %1, %2;" : "=r"(sub_res[0]) : "r"(tmp[0]), "r"(KH_P_at(0)));
    #pragma unroll
    for (int i = 1; i < 8; i++) {
      asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res[i]) : "r"(tmp[i]), "r"(KH_P_at(i)));
    }
    asm("subc.u32 %0, %1, 0;" : "=r"(borrow) : "r"(even[8]));

    // 如果没有借位 (borrow == 0)，说明 tmp >= p，使用 sub_res；否则使用 tmp
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      res[i] = (borrow == 0) ? sub_res[i] : tmp[i];
    }

  #else
    // 非 CUDA 设备回退到 C 实现
    mont_mul_c(res, a, b);
  #endif
  }

  // 统一入口：暂时回退到 C 版本 Montgomery（待修复）
  __device__ __forceinline__ static void mont_mul(uint32_t res[8], const uint32_t a[8], const uint32_t b[8]) {
    mont_mul_c(res, a, b);
  }

  // Montgomery 转换函数
  __device__ __forceinline__ Fp &inplace_to_montgomery() {
    Fp r2;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
#if defined(__CUDA_ARCH__)
      r2.d[i] = KH_R2_at(i);
#else
      r2.d[i] = Params::R2[i];
#endif
    }
    Fp result;
    mont_mul(result.d, d, r2.d);
    *this = result;
    return *this;
  }

  __device__ __forceinline__ Fp from_montgomery() const {
    Fp one{}; one.d[0] = 1u;
    Fp result;
    mont_mul(result.d, d, one.d);
    return result;
  }

  // Arithmetic
  __device__ __forceinline__ Fp operator+(const Fp &o) const {
    Fp r; add_n(r.d, d, o.d); return r; 
  }
  __device__ __forceinline__ Fp operator-(const Fp &o) const {
    Fp r; sub_n(r.d, d, o.d); return r; 
  }
  __device__ __forceinline__ Fp operator*(const Fp &o) const {
    Fp r; mont_mul(r.d, d, o.d); return r; 
  }
  __device__ __forceinline__ Fp square() const { return (*this) * (*this); }
  __device__ __forceinline__ bool is_zero() const {
    uint32_t acc = 0u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc |= d[i];
    return acc == 0u;
  }
  __device__ __forceinline__ bool operator==(const Fp &o) const {
    uint32_t acc = 0u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc |= (d[i] ^ o.d[i]);
    return acc == 0u;
  }

  // Multiplicative inverse in Fp using exponentiation (a^(p-2) mod p).
  __device__ __forceinline__ Fp inverse() const {
    if (is_zero()) { return *this; }
    // Exponent e = p - 2 (little-endian limbs)
    uint32_t e[8];
    // e = P - 2
    uint64_t borrow = 2;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint64_t pi = (uint64_t)(
#if defined(__CUDA_ARCH__)
        KH_P_at(i)
#else
        Params::P[i]
#endif
      );
      uint64_t t = pi - (borrow & 0xFFFFFFFFull);
      e[i] = (uint32_t)t;
      borrow = (t >> 63) & 1ull; // 1 if underflow
    }

    Fp base = *this;         // Montgomery domain
    Fp result = mont_one();  // Montgomery 1

    for (int i = 7; i >= 0; --i) {
      uint32_t w = e[i];
      for (int b = 31; b >= 0; --b) {
        result = result.square();
        if ((w >> b) & 1u) {
          result = result * base;
        }
      }
    }
    return result;
  }
};

} // namespace secp256k1
} // namespace kh_ecc
