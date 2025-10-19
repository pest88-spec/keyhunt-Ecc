# Stage 4 CUDA Kernel Resource & Efficiency Analysis

_Last updated: 2025-10-18_

## 1. Tooling status

The Stage 4 ticket requires building each CUDA translation unit with
`nvcc --ptxas-options=-v --resource-usage` and sampling representative workloads under
`nvprof`/Nsight Compute. The current CI VM does not ship the NVIDIA CUDA
Toolkit, so both `nvcc` and `nvprof` are unavailable:

```bash
$ nvcc --version
bash: nvcc: command not found

$ nvprof --version
bash: nvprof: command not found
```

Because of this, the `.resource` reports and ptxas register/spill metrics could
not be generated inside the VM. The commands to collect them once a CUDA
installation is present are documented in §2. Static analysis of the kernels and
launch sites is provided in §§3–4.

## 2. Reproduction checklist for hardware-enabled environments

Run the following on a host that has the CUDA toolkit (11.x+) and an NVIDIA GPU:

```bash
# 1. Configure a cmake build that enables resource reporting on every compilation
cmake -S . -B build-stage4 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_FLAGS="--ptxas-options=-v --resource-usage"

# 2. Build the CUDA objects (this emits *.resource files next to each object)
cmake --build build-stage4 --target all

# 3. Archive the resource reports
find build-stage4 -name '*.resource' -exec cp {} dev-support/reports/stage4/ptxas/ \;

# 4. Profile representative workloads (requires an attached GPU)
# Example: run the gtest binary that exercises the ECDSA batch kernels
ncu --set full --target-processes all --export dev-support/reports/stage4/ncu \
  build-stage4/test/ecdsa_sign_bk3_test
```

Any dataset collected via the above commands can be committed under
`dev-support/reports/stage4/` alongside this memo.

A helper script is provided to automate collection when CUDA tools are available:

```bash
# From the repo root
bash scripts/stage4_profile.sh
```

## 3. Kernel resource snapshot (static inspection)

| Kernel | Location | Dynamic shared memory | Observations | Suggested actions |
| --- | --- | --- | --- | --- |
| `keyhunt_batch_pmul` | `KEYHUNT-ECC/core/batch_kernel.h` | `2 * blockDim.x * sizeof(Fp)` → 64 bytes/thread (Fp = 8×u32) → 16 KiB at 256 threads | Per-thread Jacobian precompute `ECPointJacobian J[16]` and numerous temporaries will push register count well above 128 ⇒ likely occupancy bottleneck and spills. Serial `montgomery_batch_inverse` (executed by thread 0) scales poorly with block size. | Move window table to shared memory or use warp-level cooperative precomputation, shrink live ranges (split kernel or stage precompute/ladder), and parallelise the batch inversion (warp-prefix scan or segmented scans). Consider `__launch_bounds__(256, 3)` with explicit register capping once register counts are known. |
| `processScalarKey` / `processScalarPoint` | `include/gecc/arith/batch_ec.h` | none | Uses `load_arbitrary` paths that bounce between SoA/AoS layouts without coalescing guarantees; repeated Montgomery conversions cause extra ALU + register pressure. | Specialise for the selected layout and hoist `inplace_to_montgomery()` to a preprocessing pass so the kernel runs purely in Montgomery domain. |
| `scalarMulByCombinedDAA` | `include/gecc/arith/batch_ec.h` | `blockDim.x * Layout::WIDTH * sizeof(BaseField)` (≈ 256 × 8 × 32 bytes = 64 KiB for current layout) | Shared-memory footprint caps blocks-per-SM to 1–2 on SM80 chips. The forward/backward traversal performs per-slot prefix scans with heavy use of `__syncthreads()` inside the outer scalar loop, so latency scales with `Fr::BITS`. | Reduce `Layout::WIDTH` for large batches, or split kernel into precompute + accumulation phases so the shared buffer can be double-buffered and reuse `__shfl_sync` to avoid serialising on shared memory. |
| `fixedPMulByCombinedDAA` | `include/gecc/arith/batch_ec.h` | Same as above | Similar synchronisation patterns and shared-memory pressure. The kernel reloads points from global memory twice per bit pass and writes back on every iteration, hinting at uncoalesced traffic when `GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS` is disabled. | Introduce SoA layout for point buffers unconditionally, keep points resident in registers during the inner loop, and only spill to global memory after batch updates. |
| `TestFp` / `TestFp_speed` | `test/fp.cu` | none | Unit kernels used for correctness/benchmarking allocate managed memory and launch with a single block. The benchmark variant performs 10 000 dependent multiplies in-register, so profiling it under Nsight will show 0% occupancy for compute-bound loops but emphasises register pressure. Managed allocations also hide host/device copy costs. | For profiling, convert to pinned host buffers + explicit `cudaMemcpyAsync` to surface transfer cost, and cap the inner loop to keep runtime short while still stressing arithmetic pipelines. |

> **Note**: Register counts and spill metrics will be filled in once `nvcc` is
> available and the `.resource` artifacts are harvested.

## 4. Launch configuration audit

* `KEYHUNT-ECC/api/bridge.cu` launches `keyhunt_batch_pmul` with caller-specified
  `block_dim` but always allocates shared memory as if the block were full
  (`2 * blockDim.x * sizeof(Fp)`), so blocks that process partial tail tiles still
  reserve the full 16 KiB. Consider computing the shared-memory size using
  `min(blockDim.x, local_count)`.
* `cuda/launch_config.h` estimates occupancy using
  `max_shared_memory_per_block / shared_memory_size`, ignoring register limits and
  the max blocks-per-SM restriction on recent SMs. Replacing the home-grown
  heuristic with `cudaOccupancyMaxPotentialBlockSize` (or at least intersecting
  shared-memory, register, and thread budgets) will produce more accurate grid
  sizing.
* Cooperative and SoA variants in the bridge currently return
  `cudaErrorNotYetImplemented`; they still compute (and reserve) heavier shared
  memory footprints in their placeholder comments (up to ~48 KiB/block). Leave a
  note for when those variants are enabled so the occupancy calculation can be
  revisited.

## 5. Host/device transfer patterns

* The keyhunt GPU backend copies the `x` and `y` coordinate buffers in two
  separate `cudaMemcpy` calls. If device bandwidth is the bottleneck, fusing them
  through a single contiguous buffer (or using vectorised loads/stores inside the
  kernel) reduces PCIe transactions.
* `GPU_BatchPrivToPub_Bytes32BE` allocates and frees three host buffers on every
  call. Pooling these buffers (mirroring the device pools) or using stack
  scratchpads dramatically lowers host-side churn.
* All unit tests use `cudaMallocManaged`; replacing this with explicit page-locked
  host buffers exposes realistic transfer timings during profiling.

## 6. Recommended next steps

1. Install the CUDA toolkit on the profiling host, rerun the commands in §2 (or `scripts/stage4_profile.sh`), and
   add the generated `.resource` and Nsight reports under
   `dev-support/reports/stage4/`.
2. Once register counts are known, evaluate the impact of
   `__launch_bounds__`/`-maxrregcount` on `keyhunt_batch_pmul` and consider
   splitting the window precomputation out of the hot kernel if occupancy is
   still <50%.
3. Prototype a warp-parallel batch inversion (e.g., using segmented scans) to
   replace the serial section guarded by `tid == 0`.
4. Validate the launch heuristics by calling `cudaOccupancyAvailableDynamicSMemPerBlock`
   with the calculated shared-memory footprint and compare against actual Nsight
   Compute occupancy metrics.

## 7. Atomics and memory-access notes

- Static inspection (grep) found no CUDA device atomics in the `KEYHUNT-ECC` kernels or test `.cu` sources; contention from atomics is not a present bottleneck.
- Risk for uncoalesced reads/writes arises mainly from AoS layouts when `GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS` is disabled; prefer SoA for batch kernels and vectorised stores on output.

Once the toolkit is available, append the measured register/shared-mem usage and
occupancy data to the table in §3.
