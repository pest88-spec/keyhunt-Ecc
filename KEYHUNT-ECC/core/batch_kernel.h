#pragma once
#include <stdint.h>
#include "../cuda/device_functions.h"
#include "ec_point.h"

namespace kh_ecc {
namespace secp256k1 {

// 批量链式逆元（占位）：前缀乘积 + 单次逆元 + 回传恢复个体逆
// elems[i] 输入为非零场元（Montgomery 域），输出覆盖为其逆元。
__device__ __forceinline__ void montgomery_batch_inverse(Fp* elems, uint32_t count) {
  if (count == 0u) return;
  if (count == 1u) { elems[0] = elems[0].inverse(); return; }
  // 简单的就地前缀法需要额外 scratch；提供重载以传入 scratch
}

// 重载：带 scratch 缓冲（长度 = count）
__device__ __forceinline__ void montgomery_batch_inverse(Fp* elems, Fp* scratch, uint32_t count) {
  if (count == 0u) return;
  if (count == 1u) { elems[0] = elems[0].inverse(); return; }
  // 前缀乘积
  scratch[0] = elems[0];
  for (uint32_t i = 1; i < count; ++i) {
    scratch[i] = scratch[i-1] * elems[i];
  }
  // 计算总积的逆
  Fp inv_total = scratch[count - 1].inverse();
  // 反向回传，注意在写回前保存原值
  for (int i = (int)count - 1; i >= 1; --i) {
    Fp orig = elems[i];
    elems[i] = inv_total * scratch[i-1];
    inv_total = inv_total * orig;
  }
  // i == 0
  elems[0] = inv_total;
}

// Batch private_key * G (secp256k1) kernel interface (skeleton)
// Memory layout:
// - private_keys: count x 8 limbs (uint32 LE) of scalars in [0, n)
// - public_keys_x/y: output arrays, each count x 8 limbs (uint32 LE)
// - count: number of scalars
__global__ void keyhunt_batch_pmul(
    const uint32_t* __restrict__ private_keys,
    uint32_t* __restrict__ public_keys_x,
    uint32_t* __restrict__ public_keys_y,
    uint32_t count
) {
  const uint32_t block_base = blockIdx.x * blockDim.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t instance = block_base + tid;

  // 局部激活线程数（最后一个 block 可能不足）
  uint32_t local_count = 0;
  if (block_base < count) {
    uint32_t remain = count - block_base;
    local_count = remain < blockDim.x ? remain : blockDim.x;
  }
  const bool active = (tid < local_count);

  // 动态共享内存布局：zarr[local_count] | scratch[local_count]
  extern __shared__ unsigned char smem[];
  Fp* zarr    = reinterpret_cast<Fp*>(smem);
  Fp* scratch = reinterpret_cast<Fp*>(zarr + blockDim.x);

  // 预备：基点 G（Affine，Montgomery 域）
  Affine G;
  {
    Fp gx, gy;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
#if defined(__CUDA_ARCH__)
      // 直接内联常量，避免在设备代码中访问类内 constexpr 数组的限制
      const uint32_t Gx_i = (i==0)?0x16F81798u:(i==1)?0x59F2815Bu:(i==2)?0x2DCE28D9u:(i==3)?0x029BFCDBu:(i==4)?0xCE870B07u:(i==5)?0x55A06295u:(i==6)?0xF9DCBBACu:0x79BE667Eu;
      const uint32_t Gy_i = (i==0)?0xFB10D4B8u:(i==1)?0x9C47D08Fu:(i==2)?0xA6855419u:(i==3)?0xFD17B448u:(i==4)?0x0E1108A8u:(i==5)?0x5DA4FBFCu:(i==6)?0x26A3C465u:0x483ADA77u;
      gx.d[i] = Gx_i; gy.d[i] = Gy_i;
#else
      gx.d[i] = Params::Gx[i]; gy.d[i] = Params::Gy[i];
#endif
    }
    gx.inplace_to_montgomery();
    gy.inplace_to_montgomery();
    G.x = gx; G.y = gy;
  }

  // 每线程：读取私钥 k，使用固定窗口 w=4 的预计算 + 窗口双倍-加法
  ECPointJacobian P = ECPointJacobian::zero();
  if (active) {
    uint32_t k[8];
    // little-endian 读取 8 limbs
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      k[i] = private_keys[instance * 8 + i];
    }

    // 检查 k 是否为 0
    uint32_t acc_or = 0u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc_or |= k[i];
    const bool k_is_zero = (acc_or == 0u);

    if (!k_is_zero) {
      // 预计算 J[1..15]（Jacobian），J[0] 不用
      const int W = 4;
      ECPointJacobian J[1 << W];
      // J[1] = G（Jacobian）
      J[1].x = G.x; J[1].y = G.y; J[1].z = Fp::mont_one();
      for (int v = 2; v < (1 << W); ++v) {
        // J[v] = J[v-1] + G（混合加）
        J[v] = J[v-1] + G; // mixed_add
      }

      // 总共有 256 / W = 64 个窗口，从高到低
      for (int widx = (256 / W) - 1; widx >= 0; --widx) {
        // W 次倍加
        #pragma unroll
        for (int d = 0; d < W; ++d) P = P.dbl();
        // 取该窗口的 nibble 值（小端）
        int bitpos = widx * W;           // 0..252
        int limb_i = bitpos / 32;        // 0..7（按小端）
        int off    = bitpos % 32;        // 0..28
        uint32_t limb = k[limb_i];
        uint32_t v = (limb >> off) & 0xFu;
        // 注意跨 limb 情况：当 off > 28 时会跨界，这里 W=4，off ∈ {0,4,8,...,28} 不会跨界
        if (v) {
          P = P + J[v]; // Jacobian + Jacobian
        }
      }
    }
  }

  // 处理 k=0 的情况：P.z 为 0，不能参与批量逆元
  bool has_valid_z = false;
  if (active) {
    uint32_t z_acc = 0u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) z_acc |= P.z.d[i];
    has_valid_z = (z_acc != 0u);
    
    // 对于有效 z，存入 zarr；否则存 mont_one（避免除零）
    zarr[tid] = has_valid_z ? P.z : Fp::mont_one();
  }
  __syncthreads();

  if (tid == 0 && local_count > 0) {
    montgomery_batch_inverse(zarr, scratch, local_count);
  }
  __syncthreads();

  if (active) {
    if (has_valid_z) {
      // 正常情况：使用批量逆元结果
      Fp inv_z = zarr[tid];
      Fp inv_z2 = inv_z.square();
      Fp inv_z3 = inv_z2 * inv_z;

      Fp x_m = P.x * inv_z2;
      Fp y_m = P.y * inv_z3;
      
      Fp x_n = x_m.from_montgomery();
      Fp y_n = y_m.from_montgomery();

      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        public_keys_x[instance * 8 + i] = x_n.d[i];
        public_keys_y[instance * 8 + i] = y_n.d[i];
      }
    } else {
      // k=0 的情况：输出全零坐标
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        public_keys_x[instance * 8 + i] = 0u;
        public_keys_y[instance * 8 + i] = 0u;
      }
    }
  }
}

} // namespace secp256k1
} // namespace kh_ecc
