#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "gpu_backend.h"

// 临时存根实现，用于验证集成
extern "C" int GPU_IsAvailable() {
  // 返回0表示GPU不可用，将使用CPU模式
  return 0;
}

extern "C" int GPU_BatchPrivToPub(const uint32_t* h_private_keys,
                                   uint32_t* h_public_keys_x,
                                   uint32_t* h_public_keys_y,
                                   uint32_t count,
                                   uint32_t block_dim) {
  // 临时存根实现，返回错误表示GPU不可用
  return -1;
}

