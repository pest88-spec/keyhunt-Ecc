# Stage 2 Static Audit – Placeholders, Stubs, and Mock Implementations

_Generated 2025-10-18 03:27:50Z via `stage2_static_audit.py`._

This report aggregates repository-wide scans for placeholder markers,
potential stub implementations, and conditional branches that may gate
mock or fallback code paths. Findings are grouped by category for
downstream review.


## 1. Marker Sweep

Total marker hits: 30

### FIXME (3)
- `albertobsd-keyhunt/gmp256k1/GMP256K1.cpp:642` — fprintf(stderr,"[E] Fixme unsopported case");
- `albertobsd-keyhunt/secp256k1/SECP256K1.cpp:805` — fprintf(stderr,"[E] Fixme unsopported case");
- `albertobsd-keyhunt/xxhash/xxhash.h:3527` — * FIXME: Clang's output is still _much_ faster -- On an AMD Ryzen 3600,

### HACK (16)
- `albertobsd-keyhunt/xxhash/xxhash.h:1739` — * UGLY HACK:
- `albertobsd-keyhunt/xxhash/xxhash.h:1740` — * This inline assembly hack forces acc into a normal register. This is the
- `albertobsd-keyhunt/xxhash/xxhash.h:1768` — * How this hack works:
- `albertobsd-keyhunt/xxhash/xxhash.h:2901` — * UGLY HACK:
- `albertobsd-keyhunt/xxhash/xxhash.h:3518` — && !defined(XXH_ENABLE_AUTOVECTORIZE)      /* Define to disable like XXH32 hack */
- `albertobsd-keyhunt/xxhash/xxhash.h:3520` — * UGLY HACK:
- `albertobsd-keyhunt/xxhash/xxhash.h:3599` — * UGLY HACK:
- `albertobsd-keyhunt/xxhash/xxhash.h:3878` — * The asm hack causes Clang to assume that XXH3_kSecretPtr aliases with
- `albertobsd-keyhunt/xxhash/xxhash.h:4202` — * We need a separate pointer for the hack below,
- `albertobsd-keyhunt/xxhash/xxhash.h:4211` — * UGLY HACK:
- `albertobsd-keyhunt/xxhash/xxhash.h:4235` — *   without hack: 2654.4 MB/s
- `albertobsd-keyhunt/xxhash/xxhash.h:4236` — *   with hack:    3202.9 MB/s
- `albertobsd-keyhunt/xxhash/xxhash.h:4250` — * The asm hack causes Clang to assume that kSecretPtr aliases with
- `albertobsd-keyhunt/xxhash/xxhash.h:4396` — * UGLY HACK:
- `albertobsd-keyhunt/xxhash/xxhash.h:4400` — *   without hack: 2063.7 MB/s
- `albertobsd-keyhunt/xxhash/xxhash.h:4401` — *   with hack:    2560.7 MB/s

### TEMPORARY (1)
- `KEYHUNT-ECC/core/fp_montgomery_new.h:2` — // Temporary file for testing

### TODO (10)
- `KEYHUNT-ECC/core/fp_bitcrack.h:282` — // TODO: 完整的 BitCrack mulModP 实现
- `albertobsd-keyhunt/hash/sha512.cpp:367` — // TODO Handle key larger than 128
- `albertobsd-keyhunt/secp256k1/Int.cpp:1025` — // TODO: compute max digit
- `albertobsd-keyhunt/xxhash/xxhash.h:41` — /* TODO: update */
- `albertobsd-keyhunt/xxhash/xxhash.h:2864` — && defined(__GNUC__) /* TODO: IBM XL */
- `include/common.h:27` — // TODO: support blockDim > 65536
- `include/gecc/arith/digit.h:78` — // TODO: Improve
- `include/gecc/arith/ec.h:638` — // TODO: not support for BLS12_377 curve
- `include/gecc/common.h:38` — // TODO: support blockDim > 65536
- `test/ecdsa_sign_baseline.cu:96` — // TODO OPT


## 2. Potential Stub Implementations

No potential stub implementations detected.


## 3. Conditional Compilation / Mock Paths

Total conditional branches: 24

### GPU (21)
- `KEYHUNT-ECC/core/batch_kernel.h:72`
  
    #if defined(__CUDA_ARCH__)
          // 直接内联常量，避免在设备代码中访问类内 constexpr 数组的限制
          const uint32_t Gx_i = (i==0)?0x16F81798u:(i==1)?0x59F2815Bu:(i==2)?0x2DCE28D9u:(i==3)?0x029BFCDBu:(i==4)?0xCE870B07u:(i==5)?0x55A06295u:(i==6)?0xF9DCBBACu:0x79BE667Eu;
          const uint32_t Gy_i = (i==0)?0xFB10D4B8u:(i==1)?0x9C47D08Fu:(i==2)?0xA6855419u:(i==3)?0xFD17B448u:(i==4)?0x0E1108A8u:(i==5)?0x5DA4FBFCu:(i==6)?0x26A3C465u:0x483ADA77u;

- `KEYHUNT-ECC/core/fp_bitcrack.h:30`
  
    #if defined(__CUDA_ARCH__)

- `KEYHUNT-ECC/core/fp_bitcrack.h:84`
  
    #if defined(__CUDA_ARCH__)
        // Little-endian: 从索引 0（LSB）开始加
        BC_ADD_CC(c[0], a[0], b[0]);
        BC_ADDC_CC(c[1], a[1], b[1]);

- `KEYHUNT-ECC/core/fp_bitcrack.h:177`
  
    #if defined(__CUDA_ARCH__)
        // Little-endian: 从索引 0（LSB）开始减
        BC_SUB_CC(c[0], a[0], b[0]);
        BC_SUBC_CC(c[1], a[1], b[1]);

- `KEYHUNT-ECC/core/fp_bitcrack.h:248`
  
    #if defined(__CUDA_ARCH__)
        // 注意：BitCrack 原始代码使用 big-endian，需要适配为 little-endian
        // 为简化实现，我们转换为 big-endian 执行 BitCrack 算法，然后转换回来

- `KEYHUNT-ECC/core/fp_montgomery.h:11`
  
    #if defined(__CUDA_ARCH__)
    __device__ __forceinline__ uint32_t KH_P_at(int i) {
      switch(i){
        case 0:return 0xFFFFFC2Fu; case 1:return 0xFFFFFFFEu; case 2:return 0xFFFFFFFFu; case 3:return 0xFFFFFFFFu;

- `KEYHUNT-ECC/core/fp_montgomery.h:56`
  
    #if defined(__CUDA_ARCH__)
          r.d[i] = KH_R_at(i);
    #else
          r.d[i] = Params::R[i];

- `KEYHUNT-ECC/core/fp_montgomery.h:67`
  
    #if defined(__CUDA_ARCH__)
        // acc[0..n-1] = a[0..n-1] * bi (lo/hi 展开)，布局仿 gECC mul_n
        for (size_t j = 0; j < n; j += 2) {
          asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"

- `KEYHUNT-ECC/core/fp_montgomery.h:88`
  
    #if defined(__CUDA_ARCH__)
        // acc += a * bi，使用 mad*.u32 链式带进位累加
        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
            : "+r"(acc[0]), "+r"(acc[1])

- `KEYHUNT-ECC/core/fp_montgomery.h:111`
  
    #if defined(__CUDA_ARCH__)
        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
            : "+r"(acc[0]), "+r"(acc[1])
            : "r"(a[0]), "r"(bi));

- `KEYHUNT-ECC/core/fp_montgomery.h:140`
  
    #if defined(__CUDA_ARCH__)
      // Compare a >= P (device side, inline fetch of P)
      __device__ __forceinline__ static bool geq_p(const uint32_t a[8]) {
        for (int i = 7; i >= 0; --i) {

- `KEYHUNT-ECC/core/fp_montgomery.h:168`
  
    #if defined(__CUDA_ARCH__)
              KH_P_at(i)
    #else
              Params::P[i]

- `KEYHUNT-ECC/core/fp_montgomery.h:190`
  
    #if defined(__CUDA_ARCH__)
        if (carry || geq_p(c)) {
    #else
        if (carry || geq(c, Params::P)) {

- `KEYHUNT-ECC/core/fp_montgomery.h:199`
  
    #if defined(__CUDA_ARCH__)
              KH_P_at(i)
    #else
              Params::P[i]

- `KEYHUNT-ECC/core/fp_montgomery.h:214`
  
    #if defined(__CUDA_ARCH__)
          KH_PINV()
    #else
          Params::PINV

- `KEYHUNT-ECC/core/fp_montgomery.h:247`
  
    #if defined(__CUDA_ARCH__)
              KH_P_at(j)
    #else
              Params::P[j]

- `KEYHUNT-ECC/core/fp_montgomery.h:269`
  
    #if defined(__CUDA_ARCH__)
        if (t8 > 0 || geq_p(t)) {
          // t - P（普通减法，不是模减法）
          uint64_t borrow = 0;

- `KEYHUNT-ECC/core/fp_montgomery.h:309`
  
    #if defined(__CUDA_ARCH__)
        // CIOS (Coarsely Integrated Operand Scanning) Montgomery 乘法
        uint32_t even[9], odd[9];
        even[8] = 0;

- `KEYHUNT-ECC/core/fp_montgomery.h:476`
  
    #if defined(__CUDA_ARCH__)
          r2.d[i] = KH_R2_at(i);
    #else
          r2.d[i] = Params::R2[i];

- `KEYHUNT-ECC/core/fp_montgomery.h:529`
  
    #if defined(__CUDA_ARCH__)
            KH_P_at(i)
    #else
            Params::P[i]

- `KEYHUNT-ECC/cuda/device_functions.h:4`
  
    #ifndef __CUDACC__

### TEST (3)
- `albertobsd-keyhunt/xxhash/xxhash.h:2684`
  
    #if ((defined(sun) || defined(__sun)) && __cplusplus) /* Solaris includes __STDC_VERSION__ with C++. Tested with GCC 5.5 */

- `include/common.h:34`
  
    #ifndef GECC_QAPW_TEST

- `include/gecc/common.h:45`
  
    #ifndef GECC_QAPW_TEST
