#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// 返回 1 表示有可用 CUDA 设备，否则 0
int GPU_IsAvailable();

// Host 内存接口：输入私钥（count x 8×uint32 LE），输出公钥坐标（各 count x 8×uint32 LE）
// 返回 0 表示成功，非 0 为 CUDA 错误码
int GPU_BatchPrivToPub(const uint32_t* h_private_keys,
                       uint32_t* h_public_keys_x,
                       uint32_t* h_public_keys_y,
                       uint32_t count,
                       uint32_t block_dim);

// 字节序转换辅助函数：32字节大端(BE) <-> 8×uint32小端(LE)
void Convert_BE32_to_LE32_Array(const uint8_t* be_bytes, uint32_t* le_words);
void Convert_LE32_to_BE32_Array(const uint32_t* le_words, uint8_t* be_bytes);

// 大端字节流接口：输入私钥（count x 32字节 BE），输出公钥坐标（各 count x 32字节 BE）
// 返回 0 表示成功，非 0 为 CUDA 错误码
int GPU_BatchPrivToPub_Bytes32BE(const uint8_t* h_private_keys_be,
                                 uint8_t* h_public_keys_x_be,
                                 uint8_t* h_public_keys_y_be,
                                 uint32_t count,
                                 uint32_t block_dim);

#ifdef __cplusplus
}
#endif