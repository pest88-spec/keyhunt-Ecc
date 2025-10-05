#include <cuda_runtime.h>
#include <stdint.h>
#include "../core/batch_kernel.h"

using namespace kh_ecc::secp256k1;

extern "C" int kh_ecc_pmul_batch(const uint32_t* d_private_keys,
                                  uint32_t* d_public_keys_x,
                                  uint32_t* d_public_keys_y,
                                  uint32_t count,
                                  uint32_t block_dim) {
  if (block_dim == 0) block_dim = 256;
  uint32_t grid_dim = (count + block_dim - 1) / block_dim;
  size_t shared_mem = 2ull * block_dim * sizeof(Fp);

  keyhunt_batch_pmul<<<grid_dim, block_dim, shared_mem>>>(
    d_private_keys, d_public_keys_x, d_public_keys_y, count
  );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return (int)err;
  err = cudaDeviceSynchronize();
  return (int)err;
}

// SoA (列主存储) 版本实现
// 注: 优化内核尚未实现，临时返回未实现错误码
// 待后续实现 keyhunt_batch_pmul_soa 内核后启用
extern "C" int kh_ecc_pmul_batch_soa(const uint32_t* const* d_private_key_limbs,
                                      uint32_t* const* d_public_key_x_limbs,
                                      uint32_t* const* d_public_key_y_limbs,
                                      uint32_t count,
                                      uint32_t block_dim) {
  (void)d_private_key_limbs;
  (void)d_public_key_x_limbs;
  (void)d_public_key_y_limbs;
  (void)count;
  (void)block_dim;
  return (int)cudaErrorNotYetImplemented;

  /* 待实现的代码:
  if (block_dim == 0) block_dim = 256;
  uint32_t grid_dim = (count + block_dim - 1) / block_dim;
  size_t shared_mem = 2ull * block_dim * sizeof(Fp);

  keyhunt_batch_pmul_soa<<<grid_dim, block_dim, shared_mem>>>(
    d_private_key_limbs, d_public_key_x_limbs, d_public_key_y_limbs, count
  );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return (int)err;
  err = cudaDeviceSynchronize();
  return (int)err;
  */
}

// Warp协作版本实现
// 注: 优化内核尚未实现，临时返回未实现错误码
// 待后续实现 keyhunt_batch_pmul_coop 内核后启用
extern "C" int kh_ecc_pmul_batch_coop(const uint32_t* d_private_keys,
                                       uint32_t* d_public_keys_x,
                                       uint32_t* d_public_keys_y,
                                       uint32_t count,
                                       uint32_t block_dim) {
  (void)d_private_keys;
  (void)d_public_keys_x;
  (void)d_public_keys_y;
  (void)count;
  (void)block_dim;
  return (int)cudaErrorNotYetImplemented;

  /* 待实现的代码:
  if (block_dim == 0) block_dim = 256;
  uint32_t grid_dim = (count + block_dim - 1) / block_dim;

  // Warp协作版本需要更多共享内存用于warp缓存
  // 计算每个block中的warp数量
  uint32_t warps_per_block = (block_dim + 31) / 32;
  size_t shared_mem = 2ull * block_dim * sizeof(Fp)          // zarr + scratch
                     + warps_per_block * 32 * 8 * sizeof(uint32_t);  // warp_cache

  keyhunt_batch_pmul_coop<<<grid_dim, block_dim, shared_mem>>>(
    d_private_keys, d_public_keys_x, d_public_keys_y, count
  );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return (int)err;
  err = cudaDeviceSynchronize();
  return (int)err;
  */
}
