#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <string>

#include "../KEYHUNT-ECC/api/bridge.h"
#include "../KEYHUNT-ECC/cuda/launch_config.h"

using namespace std::chrono;
using namespace kh_ecc::cuda;

// 性能测试结果结构
struct BenchmarkResult {
    std::string kernel_name;
    uint32_t count;
    uint32_t block_size;
    double elapsed_ms;
    double throughput_mkeys_per_sec;
    double throughput_percentage;  // 相对于基准的百分比
};

// 生成随机私钥数据
void generate_random_private_keys(std::vector<uint32_t>& keys, uint32_t count) {
    keys.resize(count * 8);
    std::mt19937 gen(42);  // 固定种子以保证可重现性
    std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFFFF);
    
    for (uint32_t i = 0; i < count; ++i) {
        for (int j = 0; j < 8; ++j) {
            keys[i * 8 + j] = dis(gen);
        }
        // 确保私钥不为零且小于 secp256k1 的阶 n
        if (keys[i * 8 + 7] >= 0xFFFFFFEFu) keys[i * 8 + 7] = 0x7FFFFFFFu;
    }
}

// 将 AoS 数据转换为 SoA 格式
void convert_to_soa(const std::vector<uint32_t>& aos_data, 
                    std::vector<std::vector<uint32_t>>& soa_data, 
                    uint32_t count) {
    soa_data.resize(8);
    for (int limb = 0; limb < 8; ++limb) {
        soa_data[limb].resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            soa_data[limb][i] = aos_data[i * 8 + limb];
        }
    }
}

// 基准测试：标准内核
BenchmarkResult benchmark_standard_kernel(uint32_t count, uint32_t iterations = 5) {
    BenchmarkResult result;
    result.kernel_name = "Standard (AoS)";
    result.count = count;
    
    // 生成测试数据
    std::vector<uint32_t> h_private_keys, h_public_x, h_public_y;
    generate_random_private_keys(h_private_keys, count);
    h_public_x.resize(count * 8);
    h_public_y.resize(count * 8);
    
    // 分配设备内存
    uint32_t *d_private_keys, *d_public_x, *d_public_y;
    size_t data_size = count * 8 * sizeof(uint32_t);
    cudaMalloc(&d_private_keys, data_size);
    cudaMalloc(&d_public_x, data_size);
    cudaMalloc(&d_public_y, data_size);
    
    // 拷贝输入数据
    cudaMemcpy(d_private_keys, h_private_keys.data(), data_size, cudaMemcpyHostToDevice);
    
    // 获取优化配置
    LaunchConfig config = compute_standard_config(count);
    result.block_size = config.block_size;
    
    // 预热
    kh_ecc_pmul_batch(d_private_keys, d_public_x, d_public_y, count, config.block_size);
    cudaDeviceSynchronize();
    
    // 基准测试
    auto start = high_resolution_clock::now();
    for (uint32_t i = 0; i < iterations; ++i) {
        kh_ecc_pmul_batch(d_private_keys, d_public_x, d_public_y, count, config.block_size);
    }
    cudaDeviceSynchronize();
    auto end = high_resolution_clock::now();
    
    result.elapsed_ms = duration_cast<microseconds>(end - start).count() / 1000.0 / iterations;
    result.throughput_mkeys_per_sec = (double)count / (result.elapsed_ms / 1000.0) / 1000000.0;
    
    // 清理
    cudaFree(d_private_keys);
    cudaFree(d_public_x);
    cudaFree(d_public_y);
    
    return result;
}

// 基准测试：SoA 内核
BenchmarkResult benchmark_soa_kernel(uint32_t count, uint32_t iterations = 5) {
    BenchmarkResult result;
    result.kernel_name = "SoA (Column-major)";
    result.count = count;
    
    // 生成和转换测试数据
    std::vector<uint32_t> h_private_keys_aos;
    generate_random_private_keys(h_private_keys_aos, count);
    
    std::vector<std::vector<uint32_t>> h_private_soa, h_public_x_soa, h_public_y_soa;
    convert_to_soa(h_private_keys_aos, h_private_soa, count);
    h_public_x_soa.resize(8);
    h_public_y_soa.resize(8);
    for (int i = 0; i < 8; ++i) {
        h_public_x_soa[i].resize(count);
        h_public_y_soa[i].resize(count);
    }
    
    // 分配设备内存
    std::vector<uint32_t*> d_private_limbs(8), d_public_x_limbs(8), d_public_y_limbs(8);
    size_t limb_size = count * sizeof(uint32_t);
    
    for (int i = 0; i < 8; ++i) {
        cudaMalloc(&d_private_limbs[i], limb_size);
        cudaMalloc(&d_public_x_limbs[i], limb_size);
        cudaMalloc(&d_public_y_limbs[i], limb_size);
        cudaMemcpy(d_private_limbs[i], h_private_soa[i].data(), limb_size, cudaMemcpyHostToDevice);
    }
    
    // 创建设备端指针数组
    uint32_t **d_private_ptrs, **d_public_x_ptrs, **d_public_y_ptrs;
    cudaMalloc(&d_private_ptrs, 8 * sizeof(uint32_t*));
    cudaMalloc(&d_public_x_ptrs, 8 * sizeof(uint32_t*));
    cudaMalloc(&d_public_y_ptrs, 8 * sizeof(uint32_t*));
    cudaMemcpy(d_private_ptrs, d_private_limbs.data(), 8 * sizeof(uint32_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_public_x_ptrs, d_public_x_limbs.data(), 8 * sizeof(uint32_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_public_y_ptrs, d_public_y_limbs.data(), 8 * sizeof(uint32_t*), cudaMemcpyHostToDevice);
    
    // 获取优化配置
    LaunchConfig config = compute_standard_config(count);
    result.block_size = config.block_size;
    
    // 预热
    kh_ecc_pmul_batch_soa(d_private_ptrs, d_public_x_ptrs, d_public_y_ptrs, count, config.block_size);
    cudaDeviceSynchronize();
    
    // 基准测试
    auto start = high_resolution_clock::now();
    for (uint32_t i = 0; i < iterations; ++i) {
        kh_ecc_pmul_batch_soa(d_private_ptrs, d_public_x_ptrs, d_public_y_ptrs, count, config.block_size);
    }
    cudaDeviceSynchronize();
    auto end = high_resolution_clock::now();
    
    result.elapsed_ms = duration_cast<microseconds>(end - start).count() / 1000.0 / iterations;
    result.throughput_mkeys_per_sec = (double)count / (result.elapsed_ms / 1000.0) / 1000000.0;
    
    // 清理
    for (int i = 0; i < 8; ++i) {
        cudaFree(d_private_limbs[i]);
        cudaFree(d_public_x_limbs[i]);
        cudaFree(d_public_y_limbs[i]);
    }
    cudaFree(d_private_ptrs);
    cudaFree(d_public_x_ptrs);
    cudaFree(d_public_y_ptrs);
    
    return result;
}

// 基准测试：协作内核
BenchmarkResult benchmark_cooperative_kernel(uint32_t count, uint32_t iterations = 5) {
    BenchmarkResult result;
    result.kernel_name = "Cooperative (Warp)";
    result.count = count;
    
    // 生成测试数据
    std::vector<uint32_t> h_private_keys, h_public_x, h_public_y;
    generate_random_private_keys(h_private_keys, count);
    h_public_x.resize(count * 8);
    h_public_y.resize(count * 8);
    
    // 分配设备内存
    uint32_t *d_private_keys, *d_public_x, *d_public_y;
    size_t data_size = count * 8 * sizeof(uint32_t);
    cudaMalloc(&d_private_keys, data_size);
    cudaMalloc(&d_public_x, data_size);
    cudaMalloc(&d_public_y, data_size);
    
    // 拷贝输入数据
    cudaMemcpy(d_private_keys, h_private_keys.data(), data_size, cudaMemcpyHostToDevice);
    
    // 获取优化配置
    LaunchConfig config = compute_cooperative_config(count);
    result.block_size = config.block_size;
    
    // 预热
    kh_ecc_pmul_batch_coop(d_private_keys, d_public_x, d_public_y, count, config.block_size);
    cudaDeviceSynchronize();
    
    // 基准测试
    auto start = high_resolution_clock::now();
    for (uint32_t i = 0; i < iterations; ++i) {
        kh_ecc_pmul_batch_coop(d_private_keys, d_public_x, d_public_y, count, config.block_size);
    }
    cudaDeviceSynchronize();
    auto end = high_resolution_clock::now();
    
    result.elapsed_ms = duration_cast<microseconds>(end - start).count() / 1000.0 / iterations;
    result.throughput_mkeys_per_sec = (double)count / (result.elapsed_ms / 1000.0) / 1000000.0;
    
    // 清理
    cudaFree(d_private_keys);
    cudaFree(d_public_x);
    cudaFree(d_public_y);
    
    return result;
}

// 打印基准测试结果
void print_benchmark_results(const std::vector<BenchmarkResult>& results, double baseline_throughput) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << \"\\n=== Performance Benchmark Results ===\\n\";
    std::cout << std::setw(20) << \"Kernel\" << std::setw(10) << \"Count\" << std::setw(12) << \"Block Size\"
              << std::setw(12) << \"Time (ms)\" << std::setw(15) << \"Throughput\" << std::setw(15) << \"Speedup\" << \"\\n\";
    std::cout << std::setw(20) << \"\" << std::setw(10) << \"\" << std::setw(12) << \"\"
              << std::setw(12) << \"\" << std::setw(15) << \"(MKey/s)\" << std::setw(15) << \"vs Baseline\" << \"\\n\";
    std::cout << std::string(88, '-') << \"\\n\";
    
    for (const auto& result : results) {
        double speedup = result.throughput_mkeys_per_sec / baseline_throughput;
        std::cout << std::setw(20) << result.kernel_name
                  << std::setw(10) << result.count
                  << std::setw(12) << result.block_size
                  << std::setw(12) << result.elapsed_ms
                  << std::setw(15) << result.throughput_mkeys_per_sec
                  << std::setw(14) << speedup << \"x\" << \"\\n\";
    }
    std::cout << \"\\n\";
}

// 主基准测试函数
void run_comprehensive_benchmark() {
    std::cout << \"KEYHUNT-ECC Performance Benchmark\\n\";
    std::cout << \"Testing GPU batch point multiplication optimizations\\n\";
    
    // 获取 GPU 信息
    int device = 0;
    cudaSetDevice(device);
    ArchTraits traits = get_arch_traits(device);
    std::cout << \"\\nGPU: Compute Capability \" << traits.major << \".\" << traits.minor << \"\\n\";
    std::cout << \"Max threads per block: \" << traits.max_threads_per_block << \"\\n\";
    std::cout << \"Max shared memory per block: \" << traits.max_shared_memory_per_block << \" bytes\\n\";
    
    std::vector<uint32_t> test_sizes = {1024, 4096, 16384, 65536, 262144};
    
    for (uint32_t count : test_sizes) {
        std::cout << \"\\n--- Testing batch size: \" << count << \" ---\\n\";
        
        std::vector<BenchmarkResult> results;
        
        // 测试标准内核（作为基准）
        std::cout << \"Testing Standard kernel...\\n\";
        results.push_back(benchmark_standard_kernel(count));
        double baseline_throughput = results[0].throughput_mkeys_per_sec;
        
        // 测试 SoA 内核
        std::cout << \"Testing SoA kernel...\\n\";
        results.push_back(benchmark_soa_kernel(count));
        
        // 测试协作内核
        std::cout << \"Testing Cooperative kernel...\\n\";
        results.push_back(benchmark_cooperative_kernel(count));
        
        // 显示结果
        print_benchmark_results(results, baseline_throughput);
        
        // 显示调优建议
        TuningAdvice advice = get_tuning_advice(count);
        std::cout << \"Tuning advice: \" << advice.reasoning << \"\\n\";
        std::cout << \"Recommended block size: \" << advice.launch_config.block_size 
                  << \", expected occupancy: \" << std::setprecision(1) 
                  << (advice.launch_config.occupancy * 100) << \"%\\n\";
    }
}

int main() {
    // 检查 CUDA 设备
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << \"Error: No CUDA devices found!\\n\";
        return 1;
    }
    
    try {
        run_comprehensive_benchmark();
    } catch (const std::exception& e) {
        std::cerr << \"Benchmark failed: \" << e.what() << \"\\n\";
        return 1;
    }
    
    std::cout << \"\\nBenchmark completed successfully!\\n\";
    return 0;
}