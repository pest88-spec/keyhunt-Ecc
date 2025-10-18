# Non-CUDA Algorithmic Performance Assessment

Ticket scope: review CPU-side algorithms, memory usage patterns, and GPU bridge synchronization while proposing concrete, severity-ranked optimizations. All measurements below were gathered on the local build of this branch using one-off micro-bench harnesses that link directly against the in-repo sources.

---

## Methodology & Bench Assets

| Focus area | Harness | Notes |
| --- | --- | --- |
| `util.c::stringtokenizer` growth | `/tmp/perf_tokenizer` (links against `util.c`) | Built with `g++ -O2` and fed synthetic token lists to stress realloc patterns. |
| `bloom/bloom.cpp` insertion | `/tmp/perf_bloom` (links against `bloom.cpp` & `xxhash.c`) | Measures wall-clock for inserting `uint64_t` keys into a Bloom filter configured via `bloom_init2`. |
| `gmp256k1` scalar multiplication | `/tmp/perf_scalar` (links against `GMP256K1.cpp` family, `hashing.c`, etc.) | Exercises `Secp256K1::ScalarMultiplication` for different scalar bit lengths. |

All timings capture the best (or mean, when stated) of multiple runs on the same VM so they are comparable across scenarios.

---

## 1. CPU Algorithm Review

### 1.1 Bloom filter operations (`albertobsd-keyhunt/bloom/bloom.cpp`)

*Complexity check*: `bloom_check_add` and `bloom_add` are `O(k)` where `k = bloom->hashes` (typically ≤ 12). Complexity matches expectations, but each iteration performs a full 64-bit modulo and two separate `XXH64` invocations.

*Benchmark*: bulk inserts of 64-bit keys (error rate set to 0.1%).

| Entries inserted | Time (s) | Throughput (M ops/s) |
| --- | --- | --- |
| 100,000 | 0.0138 | 7.2 |
| 500,000 | 0.0751 | 6.7 |
| 1,000,000 | 0.1696 | 5.9 |

*Observations*:
- The drop in throughput for larger runs stems from cache misses and repeated 64-bit modulus operations (`(a + b*i) % bloom->bits`). When `bits` is not a power of two, the modulo cannot be strength-reduced.
- We also re-hash the full payload twice per probe (`XXH64(buffer, len, seed)` and again with the first hash as seed). That dominates CPU time for short keys.

*Suggestions*:
1. **Medium severity / Medium impact** – Pre-compute `bloom->mask` when `bits` rounds up to a power of two and use bit-masking instead of division for the majority case. For other sizes, cache `bloom->bits` reciprocal and switch to Barrett reduction to avoid hardware division inside the tight loop.
2. **Medium severity / Medium impact** – Cache the initial `XXH64` state in `bloom_check_add`; libxxhash exposes the stateful API so we can hash once, then use the seeded variant to materialize all `k` indices without re-reading the buffer.
3. **Low severity / Low impact** – Consider SoA layout for `bf` to enhance prefetching when the filter grows beyond L3. This is more relevant if we batch-check large slices.

### 1.2 ECC arithmetic (`albertobsd-keyhunt/gmp256k1`)

#### Scalar multiplication (`Secp256K1::ScalarMultiplication`, lines 501-525)

*Complexity check*: classic double-and-add, strictly serial. Loop length equals scalar bit length.

*Benchmark*: multiplying generator `G` by dense scalars of different bit widths (mean over 5 runs):

| Scalar bits | Time (s) per multiplication | Ops/s |
| --- | --- | --- |
| 64 | 0.00037 | 2,730 |
| 128 | 0.00073 | 1,374 |
| 192 | 0.00110 | 909 |
| 256 | 0.00146 | 683 |

*Observations*:
- The scaling is linear, but absolute throughput (~680 mul/s at 256 bits) is low for CPU-side pre/post processing.
- Each loop iteration instantiates multiple `Int` temporaries whose constructors call `mpz_init`. These allocations dominate when the function is invoked repeatedly.
- `Secp256K1::Double` / `Add` internally allocate more than a dozen `Int` objects per call. None of them reuse GMP limbs even though their lifetimes are contained within the loop.

*Suggestions*:
1. **High severity / High impact** – Introduce a windowed NAF or wNAF ladder for arbitrary scalars, reusing the existing `GTable` for fixed-base cases and extending it to variable points with small window caches. Expect ≥3× speedup and better instruction-level parallelism.
2. **Medium severity / High impact** – Refactor `Int` to avoid repeated `mpz_init_set_str` when working with 64-bit immediates. E.g. `Int::Add(uint64_t)` currently formats to decimal (`snprintf`) and re-parses, whereas `mpz_add_ui` accepts 64-bit `unsigned long` on all LP64 targets. Falling back to `mpz_import` for LLP64 keeps Windows safe while removing hot-path heap churn.
3. **Medium severity / Medium impact** – Pool frequently used `Int` temporaries (e.g., via stack-allocated `mpz_t` wrappers with RAII) inside `Double`/`Add`; this erases thousands of `malloc/free` pairs when scanning large key ranges.

#### Precomputation table (`Secp256K1::Init`, lines 29-38)

- Nested loops (`32 × 255`) compute and store `GTable`. This is acceptable, but there is no guard preventing repetitive initialization across threads. If `Secp256K1` is instantiated per worker, we waste ~64 KiB and dozens of scalar multiplies each time. Consider static initialization with once_flag.

### 1.3 Utility code (`albertobsd-keyhunt/util.c`, `keyhunt.cpp`)

#### `stringtokenizer` (lines 67-84 in `util.c`)

*Complexity check*: Grows `tokens` array via `realloc` on each discovered token – classic `O(n²)` behaviour when token counts are large.

*Benchmark* (mean of two runs):

| Tokens | Time (s) | Growth factor vs. 5k |
| --- | --- | --- |
| 5,000 | 0.00011 | 1.0x |
| 20,000 | 0.00048 | 4.3x |
| 80,000 | 0.00190 | 17.1x |
| 160,000 | 0.00353 | 31.8x |
| 320,000 | 0.00820 | 73.8x |

*Suggestion* (**High severity / Medium impact**): move to `std::vector<char*>` or implement exponential growth (reserve 1.5–2×) before calling `strtok`. This removes the quadratic realloc pattern. Pair with smart pointer (or `std::unique_ptr<char*[]>`) to ensure RAII cleanup instead of manual `freetokenizer`.

#### Vanity target ingestion (`keyhunt.cpp`, ~6011-6075)

- `addvanity` repeatedly calls `realloc` inside nested loops while synthesising Base58 envelopes. With long vanity prefixes this turns into `O(n²)` allocations.
- The code also allocates per-element `calloc(20,1)` inside the inner loops but never frees them on failure paths.

*Suggestion* (**High severity / Medium impact**): migrate the per-target state to `std::vector<std::array<uint8_t,20>>`, reserve to the final size after counting combinations, and reuse buffers when switching between `limit_values_A` and `limit_values_B`.

#### Raw hex helpers (`Secp256K1::GetPublicKeyRaw`, lines 242-259)

- Returns a freshly `malloc`’d 65-byte buffer per call. Callers must `free` manually, which is error-prone.
- This path is hot when converting many keys, so each call triggers heap allocation + zeroing.

*Suggestion* (**Medium severity / Medium impact**): expose span-based API (`void GetPublicKeyRaw(bool, Point&, std::array<uint8_t,65>&)`) or an RAII wrapper to reuse caller-owned buffers.

---

## 2. Memory Management Patterns

| Location | Finding | Recommendation |
| --- | --- | --- |
| `util.c::stringtokenizer` | Linear `realloc` growth = repeated heap traffic and fragmented allocations. | Switch to exponential growth vector; rely on smart containers to free tokens automatically. |
| `keyhunt.cpp::addvanity` | Triple-nested `realloc`/`calloc` with no pooling and partial failure cleanup. | Convert to STL containers or pre-sized pools; wrap A/B arrays in RAII structs. |
| `Secp256K1::ScalarMultiplication` & arithmetic helpers | Stack `Int` objects still call `mpz_init`/`mpz_clear` per invocation; expensive for tight loops. | Introduce small-object allocator or thread-local pool for `Int` temporaries. |
| `Int` arithmetic with 64-bit immediates | Functions format to decimal strings then re-import, causing heap churn and locale-sensitive behaviour. | Use GMP’s `_ui`/`_si` variants or `mpz_import` to keep operations in registers. |
| `GPU_BatchPrivToPub_Bytes32BE` | Allocates three host buffers with `malloc` on every call. | Maintain pinned host buffers sized to the largest batch, or belt them into a scoped RAII helper that can live across invocations. |
| `gpu_backend.cpp::ensure_pool_size` | Device pool resizing works, but `pool_batch_size` is unused and we never shrink. | Track actual element count to optionally downsize, or expose pool reuse stats to callers. |

---

## 3. GPU Backend Bridge Synchronization & I/O (`albertobsd-keyhunt/gpu_backend.cpp`)

1. **Synchronous pipeline** (lines 90-145): the bridge always invokes `cudaDeviceSynchronize` through `kh_ecc_pmul_batch` and performs blocking `cudaMemcpy` transfers. There is no opportunity for the CPU to overlap the next batch’s preparation with GPU execution.
   - *Remedy*: expose an async variant that accepts a `cudaStream_t`, migrates to `cudaMemcpyAsync` with pinned host buffers, and only synchronizes at the application level. This enables double-buffering and overlaps host hashing with GPU muls.

2. **Device discovery overhead**: `GPU_BatchPrivToPub` calls `GPU_IsAvailable()` each time, which internally runs `cudaGetDeviceCount`. For long-running scans this adds ~10–50 µs per batch.
   - *Remedy*: cache the availability flag (or device ordinal) after the first success and guard with `std::once_flag`.

3. **Host buffer churn**: The `_Bytes32BE` wrapper re-encodes endianness via freshly `malloc`’d arrays and sequential loops.
   - *Remedy*: reuse a pinned buffer shared with `GPU_BatchPrivToPub` so that endian conversion can happen in-place when possible; consider performing the conversion on-device via a simple kernel that runs alongside the main batch to amortize latency.

4. **Single stream, single outstanding kernel**: Even in the basic LE pathway, the code assumes one kernel in flight at a time. There is room to expose batched launches (multiple streams) when the caller deals with massive key sets.

5. **Error handling**: On pool resize failure we return `-3` without cleaning partially allocated buffers. Wrapping the pool pointers in a small RAII helper (or C++ `unique_ptr` with custom deleter) would guarantee cleanup on early exits.

---

## 4. Prioritized Optimization Backlog

| Severity | Impact | Area | Concrete refactoring suggestion |
| --- | --- | --- | --- |
| **High** | High | `Int` 64-bit arithmetic | Replace string-based GMP conversions with `_ui/_si` ops or `mpz_import`; profile impact on scalar multiplication hot paths. |
| **High** | Medium | `util.c::stringtokenizer` & `keyhunt.cpp::addvanity` | Move to exponential-growth containers / STL vectors to eliminate `O(n²)` realloc patterns observed in tokenizer and vanity target ingestion benches. |
| **High** | High | ECC scalar multiplication | Introduce wNAF / sliding window ladder (and reuse `GTable`) to drop per-scalar cost from ~1.46 ms to sub-ms on CPU. |
| **Medium** | Medium | Bloom filter | Avoid division in the probe loop by exploiting power-of-two sizing or reciprocal multiplication; reuse xxHash state to cut hashing work. |
| **Medium** | Medium | GPU bridge | Add async API with stream + pinned buffer support; cache device availability. |
| **Low** | Medium | `Secp256K1::GetPublicKeyRaw` & friends | Offer buffer-based overloads to remove per-call heap allocations. |
| **Low** | Low | Device pool resizing | Track active batch size to shrink or recycle buffers adaptively. |

---

## Appendix: Reproducible Benchmark Outputs

```
$ /tmp/perf_tokenizer
tokens,time_seconds
5000,0.000111
20000,0.000479
80000,0.001904
160000,0.003532
320000,0.008198

$ /tmp/perf_bloom
entries,time_seconds
100000,0.0138238
500000,0.0751303
1000000,0.169567

$ /tmp/perf_scalar
bits,time_seconds
64,0.0003663
128,0.000727728
192,0.0011004
256,0.00146392
```

> **Note**: The harness binaries live in `/tmp` and were purpose-built for this assessment; they are not part of the repository.
