#include <stdint.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 引用我们在 KEYHUNT-ECC 中提供的 C 接口桥
// 注: 使用相对路径,由编译器的 -I 选项指定包含目录
#include "../KEYHUNT-ECC/api/bridge.h"
#include "gpu_backend.h"

extern "C" int GPU_IsAvailable() {
  int ndev = 0;
  cudaError_t err = cudaGetDeviceCount(&ndev);
  if (err != cudaSuccess) return 0;
  return (ndev > 0) ? 1 : 0;
}

extern "C" int GPU_BatchPrivToPub(const uint32_t* h_private_keys,
                                   uint32_t* h_public_keys_x,
                                   uint32_t* h_public_keys_y,
                                   uint32_t count,
                                   uint32_t block_dim) {
  if (!h_private_keys || !h_public_keys_x || !h_public_keys_y || count == 0) return -1;
  if (!GPU_IsAvailable()) return -2;

  uint32_t *d_priv = nullptr, *d_x = nullptr, *d_y = nullptr;
  size_t bytes = (size_t)count * 8u * sizeof(uint32_t);
  cudaError_t err;

  err = cudaMalloc((void**)&d_priv, bytes); if (err != cudaSuccess) goto cleanup_err;
  err = cudaMalloc((void**)&d_x, bytes); if (err != cudaSuccess) goto cleanup_err;
  err = cudaMalloc((void**)&d_y, bytes); if (err != cudaSuccess) goto cleanup_err;

  err = cudaMemcpy(d_priv, h_private_keys, bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) goto cleanup_err;

  {
    int rc = kh_ecc_pmul_batch(d_priv, d_x, d_y, count, block_dim);
    if (rc != 0) { err = (cudaError_t)rc; goto cleanup_err; }
  }

  err = cudaMemcpy(h_public_keys_x, d_x, bytes, cudaMemcpyDeviceToHost); if (err != cudaSuccess) goto cleanup_err;
  err = cudaMemcpy(h_public_keys_y, d_y, bytes, cudaMemcpyDeviceToHost); if (err != cudaSuccess) goto cleanup_err;

  cudaFree(d_priv); cudaFree(d_x); cudaFree(d_y);
  return 0;

cleanup_err:
  if (d_priv) cudaFree(d_priv);
  if (d_x) cudaFree(d_x);
  if (d_y) cudaFree(d_y);
  return (int)err;
}

// 字节序转换：32字节大端(BE) -> 8×uint32小端(LE)
// 根据 secp256k1 的 256 位大数表示，BE 字节流的低位字节对应 LE uint32 数组的高位元素
extern "C" void Convert_BE32_to_LE32_Array(const uint8_t* be_bytes, uint32_t* le_words) {
  if (!be_bytes || !le_words) return;
  
  // 将 32 字节的大端流转换为 8 个小端 uint32
  // be_bytes[0..3] -> le_words[7] (最高位 uint32)
  // be_bytes[4..7] -> le_words[6]
  // ...
  // be_bytes[28..31] -> le_words[0] (最低位 uint32)
  for (int i = 0; i < 8; i++) {
    const uint8_t* src = &be_bytes[(7-i) * 4];  // 从高位开始取 4 字节
    le_words[i] = ((uint32_t)src[3]) | 
                  ((uint32_t)src[2] << 8) |
                  ((uint32_t)src[1] << 16) |
                  ((uint32_t)src[0] << 24);
  }
}

// 字节序转换：8×uint32小端(LE) -> 32字节大端(BE)
extern "C" void Convert_LE32_to_BE32_Array(const uint32_t* le_words, uint8_t* be_bytes) {
  if (!le_words || !be_bytes) return;
  
  // 将 8 个小端 uint32 转换为 32 字节的大端流
  // le_words[0] (最低位) -> be_bytes[28..31]
  // le_words[1] -> be_bytes[24..27]
  // ...
  // le_words[7] (最高位) -> be_bytes[0..3]
  for (int i = 0; i < 8; i++) {
    uint8_t* dst = &be_bytes[(7-i) * 4];  // 向高位写入 4 字节
    uint32_t val = le_words[i];
    dst[0] = (uint8_t)(val >> 24);
    dst[1] = (uint8_t)(val >> 16);
    dst[2] = (uint8_t)(val >> 8);
    dst[3] = (uint8_t)(val);
  }
}

// 大端字节流接口包装函数
extern "C" int GPU_BatchPrivToPub_Bytes32BE(const uint8_t* h_private_keys_be,
                                             uint8_t* h_public_keys_x_be,
                                             uint8_t* h_public_keys_y_be,
                                             uint32_t count,
                                             uint32_t block_dim) {
  if (!h_private_keys_be || !h_public_keys_x_be || !h_public_keys_y_be || count == 0) return -1;
  
  // 分配临时的小端缓冲区
  uint32_t* temp_priv_le = (uint32_t*)malloc(count * 8 * sizeof(uint32_t));
  uint32_t* temp_x_le = (uint32_t*)malloc(count * 8 * sizeof(uint32_t));
  uint32_t* temp_y_le = (uint32_t*)malloc(count * 8 * sizeof(uint32_t));
  
  if (!temp_priv_le || !temp_x_le || !temp_y_le) {
    if (temp_priv_le) free(temp_priv_le);
    if (temp_x_le) free(temp_x_le);
    if (temp_y_le) free(temp_y_le);
    return -3; // 内存分配失败
  }
  
  // 转换输入：大端字节流 -> 小端 uint32 数组
  for (uint32_t i = 0; i < count; i++) {
    Convert_BE32_to_LE32_Array(&h_private_keys_be[i * 32], &temp_priv_le[i * 8]);
  }
  
  // 调用核心 GPU 函数
  int result = GPU_BatchPrivToPub(temp_priv_le, temp_x_le, temp_y_le, count, block_dim);
  
  if (result == 0) {
    // 转换输出：小端 uint32 数组 -> 大端字节流
    for (uint32_t i = 0; i < count; i++) {
      Convert_LE32_to_BE32_Array(&temp_x_le[i * 8], &h_public_keys_x_be[i * 32]);
      Convert_LE32_to_BE32_Array(&temp_y_le[i * 8], &h_public_keys_y_be[i * 32]);
    }
  }
  
  // 清理临时缓冲区
  free(temp_priv_le);
  free(temp_x_le);
  free(temp_y_le);
  
  return result;
}
