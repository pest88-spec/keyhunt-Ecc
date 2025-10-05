#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace kh_ecc {
namespace cuda {

// GPU 架构特征常量
struct ArchTraits {
    int major;                    // 计算能力主版本号
    int minor;                    // 计算能力次版本号
    int max_threads_per_block;    // 每个 block 的最大线程数
    int warp_size;                // warp 大小
    int max_shared_memory_per_block;  // 每个 block 的最大共享内存
    int max_registers_per_block;  // 每个 block 的最大寄存器数
    int max_blocks_per_sm;        // 每个 SM 的最大 block 数
    int max_threads_per_sm;       // 每个 SM 的最大线程数
};

// 获取当前设备的架构特征
inline ArchTraits get_arch_traits(int device = 0) {
    ArchTraits traits;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    traits.major = prop.major;
    traits.minor = prop.minor;
    traits.max_threads_per_block = prop.maxThreadsPerBlock;
    traits.warp_size = prop.warpSize;
    traits.max_shared_memory_per_block = prop.sharedMemPerBlock;
    traits.max_registers_per_block = prop.regsPerBlock;
    traits.max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
    traits.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    
    return traits;
}

// 批量点乘内核的优化启动配置
struct LaunchConfig {
    uint32_t block_size;          // 线程块大小
    uint32_t grid_size;           // 网格大小
    size_t shared_memory_size;    // 共享内存大小
    float occupancy;              // 预期占用率
};

// 计算标准内核的最优启动配置
inline LaunchConfig compute_standard_config(uint32_t count, int device = 0) {
    ArchTraits traits = get_arch_traits(device);
    LaunchConfig config;
    
    // 基于架构选择最优的 block 大小
    if (traits.major >= 7) {
        // Volta/Turing/Ampere: 更大的 block 通常更好
        config.block_size = 256;
    } else if (traits.major >= 6) {
        // Pascal: 中等大小的 block
        config.block_size = 256;  
    } else {
        // Maxwell/Kepler: 较小的 block
        config.block_size = 192;
    }
    
    // 确保 block_size 是 warp_size 的倍数
    config.block_size = (config.block_size / traits.warp_size) * traits.warp_size;
    
    // 计算网格大小
    config.grid_size = (count + config.block_size - 1) / config.block_size;
    
    // 计算共享内存需求（2个 Fp 数组用于 zarr 和 scratch）
    config.shared_memory_size = 2 * config.block_size * 8 * sizeof(uint32_t);
    
    // 验证共享内存限制
    if (config.shared_memory_size > traits.max_shared_memory_per_block) {
        // 如果共享内存超限，减小 block_size
        config.block_size = traits.max_shared_memory_per_block / (2 * 8 * sizeof(uint32_t));
        config.block_size = (config.block_size / traits.warp_size) * traits.warp_size;
        config.grid_size = (count + config.block_size - 1) / config.block_size;
        config.shared_memory_size = 2 * config.block_size * 8 * sizeof(uint32_t);
    }
    
    // 估算占用率（简化计算）
    int blocks_per_sm = traits.max_shared_memory_per_block / config.shared_memory_size;
    blocks_per_sm = blocks_per_sm > traits.max_blocks_per_sm ? traits.max_blocks_per_sm : blocks_per_sm;
    config.occupancy = (float)(blocks_per_sm * config.block_size) / traits.max_threads_per_sm;
    
    return config;
}

// 计算协作内核的最优启动配置
inline LaunchConfig compute_cooperative_config(uint32_t count, int device = 0) {
    ArchTraits traits = get_arch_traits(device);
    LaunchConfig config;
    
    // 协作内核需要更多共享内存，使用较小的 block_size
    if (traits.major >= 7) {
        config.block_size = 128;  // Volta+ 可以处理更多的共享内存
    } else {
        config.block_size = 96;   // 较老架构需要更保守的配置
    }
    
    // 确保是 warp_size 的倍数
    config.block_size = (config.block_size / traits.warp_size) * traits.warp_size;
    
    // 计算网格大小
    config.grid_size = (count + config.block_size - 1) / config.block_size;
    
    // 计算共享内存需求（包括 warp 缓存）
    uint32_t warps_per_block = (config.block_size + 31) / 32;
    config.shared_memory_size = 2 * config.block_size * 8 * sizeof(uint32_t)       // zarr + scratch
                              + warps_per_block * 32 * 8 * sizeof(uint32_t);       // warp_cache
    
    // 验证共享内存限制
    while (config.shared_memory_size > traits.max_shared_memory_per_block && config.block_size >= 32) {
        config.block_size -= 32;  // 减少一个 warp
        config.grid_size = (count + config.block_size - 1) / config.block_size;
        warps_per_block = (config.block_size + 31) / 32;
        config.shared_memory_size = 2 * config.block_size * 8 * sizeof(uint32_t)
                                  + warps_per_block * 32 * 8 * sizeof(uint32_t);
    }
    
    // 估算占用率
    int blocks_per_sm = traits.max_shared_memory_per_block / config.shared_memory_size;
    blocks_per_sm = blocks_per_sm > traits.max_blocks_per_sm ? traits.max_blocks_per_sm : blocks_per_sm;
    config.occupancy = (float)(blocks_per_sm * config.block_size) / traits.max_threads_per_sm;
    
    return config;
}

// 自适应选择最佳内核版本
enum class KernelVariant {
    STANDARD,     // 标准 AoS 版本
    SOA,          // 列主存储版本
    COOPERATIVE   // warp 协作版本
};

inline KernelVariant select_best_kernel(uint32_t count, int device = 0) {
    ArchTraits traits = get_arch_traits(device);
    
    // 基于数据量和架构选择最佳内核
    if (count < 1024) {
        // 小批量：标准版本开销最小
        return KernelVariant::STANDARD;
    } else if (count < 16384) {
        // 中等批量：根据架构选择
        if (traits.major >= 7) {
            return KernelVariant::COOPERATIVE;  // 新架构受益于协作
        } else {
            return KernelVariant::SOA;          // 老架构受益于 SoA
        }
    } else {
        // 大批量：SoA 通常最优，除非是最新架构
        if (traits.major >= 8) {
            return KernelVariant::COOPERATIVE;  // Ampere+ 协作最优
        } else {
            return KernelVariant::SOA;
        }
    }
}

// 性能调优建议结构
struct TuningAdvice {
    KernelVariant recommended_kernel;
    LaunchConfig launch_config;
    const char* reasoning;
};

inline TuningAdvice get_tuning_advice(uint32_t count, int device = 0) {
    TuningAdvice advice;
    ArchTraits traits = get_arch_traits(device);
    
    advice.recommended_kernel = select_best_kernel(count, device);
    
    switch (advice.recommended_kernel) {
        case KernelVariant::STANDARD:
            advice.launch_config = compute_standard_config(count, device);
            advice.reasoning = "Small batch size, standard kernel has lowest overhead";
            break;
        case KernelVariant::SOA:
            advice.launch_config = compute_standard_config(count, device);  // SoA 使用相同配置
            advice.reasoning = "Medium-to-large batch, SoA provides better memory coalescing";
            break;
        case KernelVariant::COOPERATIVE:
            advice.launch_config = compute_cooperative_config(count, device);
            advice.reasoning = "Large batch on modern GPU, cooperative loading maximizes throughput";
            break;
    }
    
    return advice;
}

} // namespace cuda
} // namespace kh_ecc