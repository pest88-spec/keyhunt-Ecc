#!/bin/bash
echo "=================================================="
echo "KEYHUNT-ECC 1G keys/s 性能优化测试"
echo "目标: 1,000,000,000 keys/s (1G keys/s)"
echo "=================================================="

# 测试配置矩阵
BATCH_SIZES=(65536 131072 262144 524288 1048576)  # 64K ~ 1M
THREADS=(4 8 12 16 20 24)

echo "测试配置矩阵:"
echo "批量大小: ${BATCH_SIZES[*]}"
echo "线程数: ${THREADS[*]}"
echo ""

# 创建测试结果CSV文件
RESULT_FILE="gpu_performance_results.csv"
echo "BatchSize,Threads,KeysPerSec,MemoryMB,GpuUtil,PerfScore" > $RESULT_FILE

echo "开始大规模性能测试..."
echo "格式: BatchSize | Threads | Keys/s | Memory MB | GPU Util | Performance Score"
echo "=============================================================================="

for batch in "${BATCH_SIZES[@]}"; do
    for threads in "${THREADS[@]}"; do
        echo ""
        echo "🧪 测试配置: Batch=${batch}, Threads=${threads}"

        # 设置环境变量
        export GPU_BATCH_SIZE=$batch

        # 运行测试（短时间测试以获取性能数据）
        output=$(timeout 20s ./keyhunt -g -m address -f tests/66.txt -b 66 -l compress -R -q -s 5 -t $threads 2>&1)

        # 提取性能数据
        keys_per_sec=$(echo "$output" | grep -oP '\d+ keys/s' | tail -1 | grep -oP '^\d+' || echo "0")

        # 计算性能分数（相对于目标的倍数）
        if [ "$keys_per_sec" -gt 0 ]; then
            perf_score=$(echo "scale=2; $keys_per_sec / 1000000000" | bc -l 2>/dev/null || echo "0")
            echo "✅ 性能: ${keys_per_sec} keys/s (${perf_score}x 目标)"

            # 记录到CSV
            echo "$batch,$threads,$keys_per_sec,N/A,N/A,$perf_score" >> $RESULT_FILE
        else
            echo "❌ 测试失败或无性能数据"
            echo "$batch,$threads,0,N/A,N/A,0" >> $RESULT_FILE
        fi

        # 短暂休息以避免GPU过热
        sleep 2
    done
done

echo ""
echo "=============================================================================="
echo "🏆 性能测试完成！结果保存到: $RESULT_FILE"
echo ""

# 找出最佳配置
echo "📊 最佳性能配置 (Top 5):"
echo "BatchSize | Threads | Keys/s | Performance vs Target"
echo "--------------------------------------------"
sort -t',' -k6 -nr $RESULT_FILE | head -6 | while IFS=',' read batch threads keys mem util score; do
    if [ "$batch" != "BatchSize" ]; then  # 跳过标题行
        printf "%-8s | %-7s | %-9s | %.2fx\n" "$batch" "$threads" "$keys" "$score"
    fi
done

# 检查是否达到目标
echo ""
best_score=$(awk -F',' 'NR>1 && $6>max { max=$6 } END { print max }' $RESULT_FILE)
if (( $(echo "$best_score >= 1.0" | bc -l) )); then
    echo "🎯 恭喜！已达到 1G keys/s 目标！最佳性能: ${best_score}x 目标"
else
    echo "⚠️  未达到 1G keys/s 目标。当前最佳: ${best_score}x 目标"
    echo ""
    echo "💡 进一步优化建议:"
    echo "1. 测试更大的批量大小 (2M, 4M)"
    echo "2. 优化 KEYHUNT-ECC GPU kernel"
    echo "3. 使用多GPU并行"
    echo "4. 优化CPU调度策略"
fi

echo ""
echo "🔧 使用最佳配置进行长时间测试:"
best_config=$(sort -t',' -k6 -nr $RESULT_FILE | head -2 | tail -1 | cut -d',' -f1,2)
best_batch=$(echo $best_config | cut -d',' -f1)
best_threads=$(echo $best_config | cut -d',' -f2)

echo "export GPU_BATCH_SIZE=$best_batch"
echo "./keyhunt -g -m address -f tests/66.txt -b 66 -l compress -R -q -s 30 -t $best_threads"