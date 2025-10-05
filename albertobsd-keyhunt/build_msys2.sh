#!/bin/bash
# Build script for keyhunt with GPU support using MSYS2 MinGW64
# Run with: bash build_msys2.sh

echo "Building keyhunt with GPU support on Windows using MSYS2..."

# Set up MSYS2 MinGW64 environment
export PATH="/mingw64/bin:/usr/bin:$PATH"
echo "PATH set to: $PATH"

# GPU integration variables (customize if needed)
export KEYHUNT_ECC_LIBDIR="D:/mybitcoin/2/keyhunt/gECC-main/KEYHUNT-ECC/build"
export CUDA_INCDIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include"
export CUDA_LIBDIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64"

echo "KEYHUNT_ECC_LIBDIR: $KEYHUNT_ECC_LIBDIR"
echo "CUDA_INCDIR: $CUDA_INCDIR"
echo "CUDA_LIBDIR: $CUDA_LIBDIR"

# Use simple compilation flags
CFLAGS="-O2"
CXXFLAGS="-O2"

echo "Compiling object files..."

# Function to compile and check errors
compile_check() {
    echo "Compiling $1..."
    if ! $2; then
        echo "Error: Failed to compile $1"
        exit 1
    fi
}

# Clean up old object files
rm -f *.o hash/*.o

# Compile individual modules
compile_check "oldbloom" "g++ $CXXFLAGS -c oldbloom/bloom.cpp -o oldbloom.o"
compile_check "bloom" "g++ $CXXFLAGS -c bloom/bloom.cpp -o bloom.o"
compile_check "base58" "gcc $CFLAGS -c base58/base58.c -o base58.o"
compile_check "rmd160" "gcc $CFLAGS -c rmd160/rmd160.c -o rmd160.o"
compile_check "sha3" "g++ $CXXFLAGS -c sha3/sha3.c -o sha3.o"
compile_check "keccak" "g++ $CXXFLAGS -c sha3/keccak.c -o keccak.o"
compile_check "xxhash" "gcc $CFLAGS -c xxhash/xxhash.c -o xxhash.o"
compile_check "util" "g++ $CXXFLAGS -c util.c -o util.o"

# Compile secp256k1 modules
echo "Compiling secp256k1 modules..."
compile_check "Int" "g++ $CXXFLAGS -c secp256k1/Int.cpp -o Int.o"
compile_check "Point" "g++ $CXXFLAGS -c secp256k1/Point.cpp -o Point.o"
compile_check "SECP256K1" "g++ $CXXFLAGS -c secp256k1/SECP256K1.cpp -o SECP256K1.o"
compile_check "IntMod" "g++ $CXXFLAGS -c secp256k1/IntMod.cpp -o IntMod.o"
compile_check "Random" "g++ $CXXFLAGS -c secp256k1/Random.cpp -o Random.o"
compile_check "IntGroup" "g++ $CXXFLAGS -c secp256k1/IntGroup.cpp -o IntGroup.o"

# Compile hash modules
echo "Compiling hash modules..."
compile_check "ripemd160" "g++ $CXXFLAGS -c hash/ripemd160.cpp -o hash/ripemd160.o"
compile_check "sha256" "g++ $CXXFLAGS -c hash/sha256.cpp -o hash/sha256.o"
# Skip SSE optimized versions due to Windows/MSYS2 compatibility issues
echo "Skipping SSE optimized hash modules (compatibility issues)"
touch hash/ripemd160_sse.o hash/sha256_sse.o

# GPU backend
echo "Compiling GPU backend..."
compile_check "gpu_backend" "g++ $CXXFLAGS -I\"$CUDA_INCDIR\" -c gpu_backend.cpp -o gpu_backend.o"

echo "Linking final executable..."
# Link with GPU backend and KEYHUNT-ECC library (note: SSE objects are empty files)
LINK_CMD="g++ $CXXFLAGS -o keyhunt.exe keyhunt.cpp base58.o rmd160.o hash/ripemd160.o hash/sha256.o bloom.o oldbloom.o xxhash.o util.o Int.o Point.o SECP256K1.o IntMod.o Random.o IntGroup.o sha3.o keccak.o gpu_backend.o -L\"$KEYHUNT_ECC_LIBDIR\" -lkeyhunt_ecc -L\"$CUDA_LIBDIR\" -lcudart -lm -lpthread"

echo "Link command: $LINK_CMD"
if ! eval $LINK_CMD; then
    echo "Error: Linking failed"
    exit 1
fi

echo "Cleaning up object files..."
rm -f *.o hash/*.o

echo "Build completed successfully!"
if [ -f keyhunt.exe ]; then
    echo "keyhunt.exe generated."
    ls -lh keyhunt.exe
else
    echo "Error: keyhunt.exe was not generated!"
    exit 1
fi

echo "Done."