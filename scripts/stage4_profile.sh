#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
REPORT_DIR="$ROOT_DIR/dev-support/reports/stage4"
TS="$(date +%Y%m%d-%H%M%S)"
PTXAS_DIR="$REPORT_DIR/ptxas/$TS"
NCU_DIR="$REPORT_DIR/ncu/$TS"
mkdir -p "$PTXAS_DIR" "$NCU_DIR"

log() { echo "[stage4] $*"; }

have() { command -v "$1" >/dev/null 2>&1; }

if ! have cmake; then
  log "cmake not found. Please install CUDA toolkit and cmake, then rerun."
  exit 0
fi

# Enable ptxas resource usage in KEYHUNT-ECC via env var
export KH_STAGE4_RESOURCE_USAGE=1

# 1) Build the KEYHUNT-ECC static library with resource usage enabled
log "Configuring KEYHUNT-ECC ..."
cmake -S "$ROOT_DIR/KEYHUNT-ECC" -B "$ROOT_DIR/KEYHUNT-ECC/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo || true
log "Building KEYHUNT-ECC ..."
(
  set -o pipefail
  cmake --build "$ROOT_DIR/KEYHUNT-ECC/build" -j"${NPROC:-$(nproc)}" 2>&1 | tee "$PTXAS_DIR/keyhunt_ecc_build.log"
) || true

# Collect any *.resource files emitted by nvcc/ptxas
log "Collecting *.resource files ..."
find "$ROOT_DIR/KEYHUNT-ECC/build" -type f -name '*.resource' -exec cp -v {} "$PTXAS_DIR" \; || true

# 2) Optionally configure the top-level (tests) with ptxas flags
if have nvcc; then
  log "Configuring top-level tests (optional) ..."
  cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build-stage4" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_FLAGS="--ptxas-options=-v --resource-usage" || true
  log "Building tests (optional) ..."
  (
    set -o pipefail
    cmake --build "$ROOT_DIR/build-stage4" -j"${NPROC:-$(nproc)}" 2>&1 | tee "$PTXAS_DIR/tests_build.log"
  ) || true

  # Copy *.resource from test build
  find "$ROOT_DIR/build-stage4" -type f -name '*.resource' -exec cp -v {} "$PTXAS_DIR" \; || true

  # 3) Profile a representative workload with Nsight Compute if available
  if have ncu; then
    # Pick an example test binary if it exists
    if [ -x "$ROOT_DIR/build-stage4/test/ecdsa_sign_bk3_test" ]; then
      log "Profiling ecdsa_sign_bk3_test with Nsight Compute ..."
      ncu --set full --target-processes all --export "$NCU_DIR/ecdsa_sign_bk3" "$ROOT_DIR/build-stage4/test/ecdsa_sign_bk3_test" || true
    elif [ -x "$ROOT_DIR/build-stage4/test/fp_test" ]; then
      log "Profiling fp_test with Nsight Compute ..."
      ncu --set full --target-processes all --export "$NCU_DIR/fp" "$ROOT_DIR/build-stage4/test/fp_test" || true
    else
      log "No test binary found for Nsight; skipping."
    fi
  elif have nvprof; then
    log "Nsight Compute not found; using nvprof (deprecated) if possible ..."
    if [ -x "$ROOT_DIR/build-stage4/test/ecdsa_sign_bk3_test" ]; then
      nvprof --log-file "$NCU_DIR/ecdsa_sign_bk3.nvprof.log" "$ROOT_DIR/build-stage4/test/ecdsa_sign_bk3_test" || true
    fi
  else
    log "Nsight Compute / nvprof not found; profiling skipped."
  fi
else
  log "nvcc not found; skipped test build and profiling."
fi

log "Stage4 collection complete."
