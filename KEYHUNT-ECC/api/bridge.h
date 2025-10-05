#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 批量私钥->公钥（secp256k1），输入/输出皆为 8xuint32 小端
// 返回0表示成功，非0为CUDA错误码
int kh_ecc_pmul_batch(const uint32_t* d_private_keys,
                      uint32_t* d_public_keys_x,
                      uint32_t* d_public_keys_y,
                      uint32_t count,
                      uint32_t block_dim);

// SoA (列主存储) 版本：提升内存合并访问效率
// private_key_limbs[8]: 指向8个limb数组的指针数组，每个数组包含count个元素
// public_key_x/y_limbs[8]: 输出的8个limb数组指针
int kh_ecc_pmul_batch_soa(const uint32_t* const* d_private_key_limbs,
                          uint32_t* const* d_public_key_x_limbs,
                          uint32_t* const* d_public_key_y_limbs,
                          uint32_t count,
                          uint32_t block_dim);

// Warp协作版本：利用warp内线程协作优化内存访问
// 输入输出格式与标准版本相同，但内部使用协作加载优化
int kh_ecc_pmul_batch_coop(const uint32_t* d_private_keys,
                           uint32_t* d_public_keys_x,
                           uint32_t* d_public_keys_y,
                           uint32_t count,
                           uint32_t block_dim);

#ifdef __cplusplus
}
#endif
