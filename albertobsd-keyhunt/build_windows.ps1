#!/usr/bin/env powershell
# Build script for keyhunt with GPU support on Windows
# Run with: powershell -ExecutionPolicy Bypass -File build_windows.ps1

Write-Host "Building keyhunt with GPU support on Windows..." -ForegroundColor Green

# GPU integration variables (customize if needed)
$KEYHUNT_ECC_LIBDIR = "D:/mybitcoin/2/keyhunt/gECC-main/KEYHUNT-ECC/build"
$CUDA_LIBDIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/lib/x64"

# Compiler paths
$GCC = "C:\msys64\mingw64\bin\gcc.exe"
$GPP = "C:\msys64\mingw64\bin\g++.exe"

Write-Host "Using GCC: $GCC" -ForegroundColor Yellow
Write-Host "Using G++: $GPP" -ForegroundColor Yellow

function Compile-Source {
    param([string]$Command)
    Write-Host "Executing: $Command" -ForegroundColor Yellow
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Compilation failed with exit code $LASTEXITCODE"
        exit 1
    }
}

Write-Host "Compiling object files..." -ForegroundColor Cyan

# Compile individual modules
Compile-Source "& '$GPP' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Wno-deprecated-copy -Ofast -ftree-vectorize -flto -c oldbloom/bloom.cpp -o oldbloom.o"
Compile-Source "& '$GPP' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Wno-deprecated-copy -Ofast -ftree-vectorize -flto -c bloom/bloom.cpp -o bloom.o"
Compile-Source "& '$GCC' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Wno-unused-parameter -Ofast -ftree-vectorize -c base58/base58.c -o base58.o"
Compile-Source "& '$GCC' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Ofast -ftree-vectorize -c rmd160/rmd160.c -o rmd160.o"
Compile-Source "& '$GPP' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Wno-deprecated-copy -Ofast -ftree-vectorize -c sha3/sha3.c -o sha3.o"
Compile-Source "& '$GPP' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Wno-deprecated-copy -Ofast -ftree-vectorize -c sha3/keccak.c -o keccak.o"
Compile-Source "& '$GCC' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Ofast -ftree-vectorize -c xxhash/xxhash.c -o xxhash.o"
Compile-Source "& '$GPP' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Wno-deprecated-copy -Ofast -ftree-vectorize -c util.c -o util.o"

# Compile secp256k1 modules
Compile-Source "& '$GPP' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Wno-deprecated-copy -Ofast -ftree-vectorize -c secp256k1/Int.cpp -o Int.o"
Compile-Source "& '$GPP' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Wno-deprecated-copy -Ofast -ftree-vectorize -c secp256k1/Point.cpp -o Point.o"
Compile-Source "& '$GPP' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Wno-deprecated-copy -Ofast -ftree-vectorize -c secp256k1/SECP256K1.cpp -o SECP256K1.o"
Compile-Source "& '$GPP' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Wno-deprecated-copy -Ofast -ftree-vectorize -c secp256k1/IntMod.cpp -o IntMod.o"
Compile-Source "& '$GPP' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Wno-deprecated-copy -Ofast -ftree-vectorize -flto -c secp256k1/Random.cpp -o Random.o"
Compile-Source "& '$GPP' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Wno-deprecated-copy -Ofast -ftree-vectorize -flto -c secp256k1/IntGroup.cpp -o IntGroup.o"

# Compile hash modules
Compile-Source "& '$GPP' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Wno-deprecated-copy -Ofast -o hash/ripemd160.o -ftree-vectorize -flto -c hash/ripemd160.cpp"
Compile-Source "& '$GPP' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Wno-deprecated-copy -Ofast -o hash/sha256.o -ftree-vectorize -flto -c hash/sha256.cpp"
Compile-Source "& '$GPP' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Wno-deprecated-copy -Ofast -o hash/ripemd160_sse.o -ftree-vectorize -flto -c hash/ripemd160_sse.cpp"
Compile-Source "& '$GPP' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Wno-deprecated-copy -Ofast -o hash/sha256_sse.o -ftree-vectorize -flto -c hash/sha256_sse.cpp"

# GPU backend
Write-Host "Compiling GPU backend..." -ForegroundColor Cyan
Compile-Source "& '$GPP' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Ofast -ftree-vectorize -flto -c gpu_backend.cpp -o gpu_backend.o"

Write-Host "Linking final executable..." -ForegroundColor Cyan
# Link with GPU backend and KEYHUNT-ECC library
$linkCommand = "& '$GPP' -m64 -march=native -mtune=native -mssse3 -Wall -Wextra -Wno-deprecated-copy -Ofast -ftree-vectorize -o keyhunt.exe keyhunt.cpp base58.o rmd160.o hash/ripemd160.o hash/ripemd160_sse.o hash/sha256.o hash/sha256_sse.o bloom.o oldbloom.o xxhash.o util.o Int.o Point.o SECP256K1.o IntMod.o Random.o IntGroup.o sha3.o keccak.o gpu_backend.o -L`"$KEYHUNT_ECC_LIBDIR`" -lkeyhunt_ecc -L`"$CUDA_LIBDIR`" -lcudart -lm -lpthread"
Compile-Source $linkCommand

Write-Host "Cleaning up object files..." -ForegroundColor Cyan
Remove-Item -Path "*.o" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "hash\*.o" -Force -ErrorAction SilentlyContinue

Write-Host "Build completed successfully!" -ForegroundColor Green
Write-Host "keyhunt.exe generated." -ForegroundColor Green

# Verify the executable was created
if (Test-Path "keyhunt.exe") {
    $fileInfo = Get-Item "keyhunt.exe"
    Write-Host "Executable size: $($fileInfo.Length) bytes" -ForegroundColor Green
    Write-Host "Created: $($fileInfo.CreationTime)" -ForegroundColor Green
} else {
    Write-Error "keyhunt.exe was not generated!"
    exit 1
}

Write-Host "Done." -ForegroundColor Green