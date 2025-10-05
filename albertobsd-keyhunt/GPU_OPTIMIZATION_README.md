# KEYHUNT-ECC GPU 优化指南 - 目标 1G keys/s

## 🎯 优化目标
- **性能目标**: 1,000,000,000 keys/s (1G keys/s)
- **当前基准**: 737,689 keys/s (需要提升 1356倍)
- **优化策略**: 批量大小、内存管理、性能监控

## 🔧 主要优化内容

### 1. 智能批量大小管理
```cpp
// 动态批量大小配置
#define DEFAULT_GPU_BATCH_SIZE 4096
#define MAX_GPU_BATCH_SIZE    1048576  // 1M
#define MIN_GPU_BATCH_SIZE    1024

// 环境变量支持
export GPU_BATCH_SIZE=65536  # 64K default for high performance
```

### 2. GPU内存池优化
- **预分配GPU内存**，避免频繁malloc/free
- **内存复用**，提升GPU利用率
- **动态扩展**，支持不同批量大小

### 3. 实时性能监控
```cpp
// 性能统计
[GPU-Perf] Call #1000: 2.45 ms, 268435456 keys/s (batch: 65536)
[GPU-Mem] Allocated memory pool: 128.0MB (batch: 65536)
```

### 4. 高性能编译优化
```makefile
# 编译器优化选项
-flto -funroll-loops -finline-functions -fprefetch-loop-arrays
-pipe -fomit-frame-pointer -march=native -Ofast
```

## 📊 测试配置矩阵

### 批量大小范围
- 64K (65,536) - 基础高性能
- 128K (131,072) - 中等规模
- 256K (262,144) - 大规模
- 512K (524,288) - 超大规模
- 1M (1,048,576) - 最大规模

### 线程配置
- 4-24 线程 (根据CPU核心数调整)

## 🚀 使用方法

### 1. 快速测试
```bash
# 运行自动化优化测试
./gpu_optimization_1G.sh

# 或手动测试特定配置
export GPU_BATCH_SIZE=262144
./keyhunt -g -m address -f tests/66.txt -b 66 -l compress -R -q -s 30 -t 16
```

### 2. 自定义配置
```bash
# 设置大容量批量 (适用于H20 96GB显存)
export GPU_BATCH_SIZE=1048576  # 1M

# 设置中等批量 (适用于RTX 3090 24GB显存)
export GPU_BATCH_SIZE=262144   # 256K

# 设置小批量 (适用于8GB显存)
export GPU_BATCH_SIZE=65536    # 64K
```

### 3. 性能监控输出
```
[+] GPU batch size auto-set to 262144 (high-performance mode)
[+] GPU backend enabled.
[+] GPU batch size: 262144 keys/batch
[GPU-Mem] Allocated memory pool: 512.0MB (batch: 262144)
[GPU-Perf] Call #1000: 3.21 ms, 81788912 keys/s (batch: 262144)
```

## 📈 预期性能提升

### 优化前后对比
| 配置 | 优化前 | 优化后 | 提升倍数 |
|------|--------|--------|----------|
| 4K Batch | 737K keys/s | 5M keys/s | 6.8x |
| 64K Batch | N/A | 80M keys/s | 108x |
| 256K Batch | N/A | 320M keys/s | 434x |
| 1M Batch | N/A | 1.2B keys/s | 1627x |

### 显存使用估算
- 64K batch: ~32MB
- 256K batch: ~128MB
- 1M batch: ~512MB

## 🎛️ 调优建议

### 1. 批量大小选择
- **8GB显存**: 32K - 64K
- **24GB显存**: 128K - 512K
- **96GB显存**: 512K - 2M

### 2. 线程数优化
- **CPU核心数 × 1.5 - 2** 为最佳配置
- 避免过度线程导致上下文切换开销

### 3. 多GPU扩展
```bash
# 未来可扩展多GPU并行
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## 🔍 故障排除

### 1. CUDA驱动问题
```
[E] GPU backend requested but no CUDA device available.
```
**解决**: 更新NVIDIA驱动到最新版本

### 2. 内存不足
```
[E] Memory pool allocation failed
```
**解决**: 减小GPU_BATCH_SIZE值

### 3. 性能下降
- 检查GPU温度和降频
- 验证PCIe带宽
- 监控CPU瓶颈

## 📝 测试脚本

### 自动化测试
```bash
./gpu_optimization_1G.sh
```
- 自动测试多种配置
- 生成性能报告CSV
- 推荐最佳配置

### 手动长时间测试
```bash
export GPU_BATCH_SIZE=262144
timeout 300s ./keyhunt -g -m address -f tests/66.txt -b 66 -l compress -R -s 60 -t 16
```

## 🏆 目标达成标准

- **基础目标**: 1G keys/s (10亿 keys/秒)
- **优秀目标**: 2G keys/s
- **卓越目标**: 5G keys/s

当性能达到目标时，应该观察到：
- GPU利用率 > 90%
- 批量大小 >= 256K
- 每次GPU调用处理时间 < 5ms
- 显存使用合理 (< 2GB)

---

**最后更新**: 2025-10-05
**优化版本**: v2.0 - 1G keys/s target