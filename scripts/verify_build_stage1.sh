#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_DIR="${ROOT_DIR}/build_logs"
MAKE_CLEAN_LOG="${LOG_DIR}/make_clean.log"
MAKE_BUILD_LOG="${LOG_DIR}/make_build.log"
CMAKE_CONFIGURE_LOG="${LOG_DIR}/cmake_configure.log"
CMAKE_BUILD_LOG="${LOG_DIR}/cmake_build.log"
SUMMARY_LOG="${LOG_DIR}/summary.txt"
CMAKE_BUILD_DIR="${ROOT_DIR}/build/stage1"

mkdir -p "${LOG_DIR}"
rm -f "${SUMMARY_LOG}"

CFLAGS_EXTRA="-Wall -Wextra"
CXXFLAGS_EXTRA="-Wall -Wextra"
CUDAFLAGS_EXTRA="-Xcompiler=-Wall,-Wextra"

run_and_log() {
  local logfile="$1"
  shift
  {
    printf '[%s] $ %s\n' "$(date --iso-8601=seconds)" "$*"
    "$@"
  } 2>&1 | tee "${logfile}"
}

pushd "${ROOT_DIR}" > /dev/null

run_and_log "${MAKE_CLEAN_LOG}" make clean

run_and_log "${MAKE_BUILD_LOG}" \
  env CFLAGS="${CFLAGS_EXTRA}" CXXFLAGS="${CXXFLAGS_EXTRA}" CUDAFLAGS="${CUDAFLAGS_EXTRA}" \
  make VERBOSE=1

rm -rf "${CMAKE_BUILD_DIR}"

run_and_log "${CMAKE_CONFIGURE_LOG}" \
  cmake -S KEYHUNT-ECC -B "${CMAKE_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="${CXXFLAGS_EXTRA}" \
    -DCMAKE_CUDA_FLAGS="${CUDAFLAGS_EXTRA}" \
    -DKEYHUNT_ECC_ENABLE_TESTS=OFF \
    -DCMAKE_VERBOSE_MAKEFILE=ON

run_and_log "${CMAKE_BUILD_LOG}" \
  cmake --build "${CMAKE_BUILD_DIR}" --config Release --verbose

popd > /dev/null

summarise_warnings() {
  local label="$1"
  local logfile="$2"
  local tmp_file
  tmp_file=$(mktemp /tmp/keyhunt_stage1_tmp.XXXXXX)
  {
    echo "## ${label}"
    if grep -iE "warning:|error:" "${logfile}" > "${tmp_file}"; then
      cat "${tmp_file}"
    else
      echo "No warnings or errors detected."
    fi
    rm -f "${tmp_file}"
    echo
  } >> "${SUMMARY_LOG}"
}

summarise_warnings "make clean" "${MAKE_CLEAN_LOG}"
summarise_warnings "make build" "${MAKE_BUILD_LOG}"
summarise_warnings "cmake configure" "${CMAKE_CONFIGURE_LOG}"
summarise_warnings "cmake build" "${CMAKE_BUILD_LOG}"

{
  echo "## Artifact overview"
  for path in \
    "${ROOT_DIR}/albertobsd-keyhunt/keyhunt" \
    "${ROOT_DIR}/KEYHUNT-ECC/build/libkeyhunt_ecc.a" \
    "${CMAKE_BUILD_DIR}/libkeyhunt_ecc.a"; do
    if [ -f "${path}" ]; then
      rel_path="${path#${ROOT_DIR}/}"
      size=$(stat --format='%s' "${path}")
      echo "${rel_path} (${size} bytes)"
    else
      rel_path="${path#${ROOT_DIR}/}"
      echo "${rel_path} (missing)"
    fi
  done
} >> "${SUMMARY_LOG}"

echo "Stage 1 build verification complete. Logs available in ${LOG_DIR}"