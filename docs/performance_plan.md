# GPU secp256k1 批量点乘 1000M key/s 优化路线图

> 目标：在保持正确性和可维护性的前提下，将 `keyhunt_batch_pmul` 内核吞吐率提升到 **≥ 1000M key/s**。本文按照“度量 → 分析 → 优化 → 验证”闭环拆解每一步的可执行动作，避免拍脑袋调参。

## 1. 建立可信的性能基线

| 维度 | 要求 | 说明 |
| --- | --- | --- |
| GPU/驱动 | 固定 GPU 型号、驱动、CUDA Toolkit 版本 | 记录 `nvidia-smi -L`、`nvidia-smi --query-driver_version` 输出 |
| 批量规模 | 4M / 16M / 64M 私钥批次 | 与 `bench.sh` 默认参数保持一致，覆盖不同缓存压力 |
| 环境监控 | 温度、功耗、频率 | 使用 `nvidia-smi --loop=1 --query-gpu=timestamp,power.draw,clocks.sm` 采样 |

执行流程：

1. 调整 `bench.sh`，增加 `--json` 输出吞吐率、执行时间、block/grid 参数，方便解析。
2. 编写 `scripts/profile_baseline.py`（待实现）：
   1. 批量运行 `bench.sh`，对三种密钥批次各执行 ≥3 次，计算平均值与标准差，若变异系数 > 5% 则自动重跑。
   2. 调用 `nv-nsight-cu-cli` 采集指标：`sm__throughput.avg.pct_of_peak_sustained_elapsed`、`dram__throughput.avg.pct_of_peak_sustained_elapsed`、`inst_executed`、`registers_per_thread`。
   3. 输出 CSV（时间戳、批量、吞吐率、SM 利用率、寄存器/线程、共享内存/块、内核持续时间）。
3. 将 CSV 归档至 `docs/perf-baseline/`，提交时保留最新一次用于回归。

## 2. 精准定位瓶颈

1. **内核级分析**：
   - `KEYHUNT-ECC/core/batch_kernel.h` 中 `keyhunt_batch_pmul` 采用固定窗口 `W=4` 的 double-and-add 路径，线程独立执行，需关注指令融合、寄存器占用与窗口表构建成本。【F:KEYHUNT-ECC/core/batch_kernel.h†L62-L135】
   - `montgomery_batch_inverse` 共享内存两次遍历，关注 `__syncthreads()` 阻塞和 `scratch` 数组的 bank conflict。【F:KEYHUNT-ECC/core/batch_kernel.h†L11-L45】
2. **调度与占用**：
   - `KEYHUNT-ECC/cuda/launch_config.h` 目前静态计算 block/grid 参数，核对是否因为 `shared_mem_bytes` 高于限制而触发 NVCC 自动降占用。【F:KEYHUNT-ECC/cuda/launch_config.h†L1-L118】
   - 用 `nvcc --ptxas-options=-v` 查看 `reg` 与 `smem`，将结果纳入 baseline 表。
3. **访存模式审计**：
   - 借助 Nsight Compute 的 `gld/gst` 效率图表确认 AoS 结构是否导致非合并访问。
   - 通过更换为 SoA（每 warp 连续加载 X、Y、Z）在试验分支验证是否提升 `dram_util`。

## 3. 算法级优化策略

| 策略 | 预期收益 | 验证方法 |
| --- | --- | --- |
| 提升窗口 `W` 或 wNAF | 减少加法轮次（理论上 `W=5` 比 `W=4` 少 ~20% 加法） | 在 `launch_config` 中暴露窗口配置，测量寄存器占用、吞吐率变化 |
| GLV 分解 | 两个 128-bit 标量并行，理论吞吐提升 1.7~1.9 倍 | 对比开启/关闭 GLV 的 Nsight profile，关注 `inst_executed` 与 `sm_efficiency` |
| 共享预计算表 | 减少每线程重复构建 `J[1<<W]` 的算术 | 统计 `shared_mem` 分配，验证是否允许更高 blockDim |
| Warp-level batch inverse | 将批量逆改写为 warp 扫描，降低共享内存压力 | 对比 `__syncthreads()` 次数与占用率变化 |

实施顺序：先实现窗口可配置与共享预计算，再评估 GLV/wNAF，最后替换批量逆算法。

## 4. CUDA 内核微结构优化

1. **寄存器与指令调优**：
   - 基于 baseline 数据锁定 reg > 128 时的路径，尝试 `__forceinline__` 函数拆分、重用 `Fp` 对象，必要时将部分中间值存入共享内存。
   - 将常用常量（如 secp256k1 模数）移入 `__constant__` 或 `constexpr`，减少加载指令。
2. **Fused 指令路径**：
   - 对 `Fp::mul`/`Fp::add` 引入内联 PTX（`mad.lo`, `mad.hi`, `add.cc`）以减少中间变量，务必通过单元测试验证结果一致。
3. **流水线化 CPU ↔ GPU**：
   - 在主机端实现三阶段 pipeline：`prepare_keys` → `cudaMemcpyAsync(H2D)` → `kernel` → `cudaMemcpyAsync(D2H)` → `verify`，分别绑定 stream0/1/2。
   - 增加 `cudaEventRecord` 度量每阶段耗时，确认 PCIe 传输不成为瓶颈。

## 5. 系统层扩展

1. **Pinned Memory 与双缓冲**：
   - 使用 `cudaHostAlloc` 为输入/输出分配 pinned buffer，构建双缓冲队列，确保数据准备与 GPU 计算重叠。
   - 通过 `nvprof --events memcpyHtoD,memcpyDtoH` 测量带宽提升。
2. **多 GPU 扩展**：
   - 每个 worker 固定一个 GPU：`cudaSetDevice(worker_id)`，使用线程安全的随机种子分片。
   - 如需跨 GPU 汇总命中结果，可引入 NCCL AllGather 或基于 ZeroMQ 的轻量 RPC。
3. **动态调参框架**：
   - 在 `KEYHUNT-ECC/cuda/launch_config.h` 暴露 `tuning_advice` 接口，将 profile 数据写入 JSON，并由启动脚本读取，按硬件自动挑选最佳 blockDim/gridDim。【F:KEYHUNT-ECC/cuda/launch_config.h†L120-L170】

## 6. 验证与回归

1. **正确性**：
   - 对比 GPU 结果与 `albertobsd-keyhunt` CPU 实现；新增随机测试样本（≥10^4），覆盖标量边界值（0、n-1、2^k）。
   - 针对 GLV/wNAF/warp batch inverse 等新路径建立独立单元测试。
2. **性能门槛**：
   - 建立 `perf_regression.sh`，若 4M 批量吞吐低于 500M key/s 或 64M 低于 750M key/s，则 CI 失败；目标 1000M key/s 作为发布前置条件。
3. **能效与热设计**：
   - 从 `nvidia-smi` 采样功耗计算 `kps_per_watt`；若出现功耗墙导致频率下降，评估限制功耗或改善散热策略。

## 7. 里程碑拆分

1. **M1：Profiling 就绪（1 周）**
   - 完成基线脚本、CSV 记录与 Nsight 指标采集。
   - 交付：`docs/perf-baseline/*.csv`、`scripts/profile_baseline.py`。
2. **M2：算法结构调整（2 周）**
   - 实现窗口可配置、共享预计算、初步 warp batch inverse，目标 ≥600M key/s。
3. **M3：寄存器/访存调优（2 周）**
   - 引入 PTX 融合算术、寄存器重排，目标 ≥800M key/s。
4. **M4：流水线 & 多 GPU（2 周）**
   - 完成 CPU/GPU 流水线化与多 GPU 协作，单卡目标 ≥900M key/s，多卡线性扩展至 1000M key/s。
5. **M5：回归体系（1 周）**
   - 将性能测试纳入 CI，形成 `perf-dashboard` 报表，确保未来改动不回退。

---

通过上述可量化的阶段性目标与度量手段，可以系统性缩小与 1000M key/s 之间的差距，并为后续优化提供可信回归依据。
