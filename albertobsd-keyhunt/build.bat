@echo off
echo Building keyhunt with GPU support on Windows...

REM GPU integration variables (customize if needed)
set KEYHUNT_ECC_LIBDIR=D:/mybitcoin/2/keyhunt/gECC-main/KEYHUNT-ECC/build
set CUDA_LIBDIR=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/lib/x64

REM Compiler paths
set GCC=C:\msys64\mingw64\bin\gcc.exe
set GPP=C:\msys64\mingw64\bin\g++.exe

echo Using GCC: %GCC%
echo Using G++: %GPP%

echo Compiling object files...

REM Use minimal compilation flags to avoid issues
set CFLAGS=-O2
set CXXFLAGS=-O2

REM Compile individual modules
echo Compiling oldbloom...
"%GPP%" %CXXFLAGS% -c oldbloom/bloom.cpp -o oldbloom.o
if errorlevel 1 goto error

echo Compiling bloom...
"%GPP%" %CXXFLAGS% -c bloom/bloom.cpp -o bloom.o
if errorlevel 1 goto error

echo Compiling base58...
"%GCC%" %CFLAGS% -c base58/base58.c -o base58.o
if errorlevel 1 goto error

echo Compiling rmd160...
"%GCC%" %CFLAGS% -c rmd160/rmd160.c -o rmd160.o
if errorlevel 1 goto error

echo Compiling sha3...
"%GPP%" %CXXFLAGS% -c sha3/sha3.c -o sha3.o
if errorlevel 1 goto error

echo Compiling keccak...
"%GPP%" %CXXFLAGS% -c sha3/keccak.c -o keccak.o
if errorlevel 1 goto error

echo Compiling xxhash...
"%GCC%" %CFLAGS% -c xxhash/xxhash.c -o xxhash.o
if errorlevel 1 goto error

echo Compiling util...
"%GPP%" %CXXFLAGS% -c util.c -o util.o
if errorlevel 1 goto error

REM Compile secp256k1 modules
echo Compiling secp256k1 modules...
"%GPP%" %CXXFLAGS% -c secp256k1/Int.cpp -o Int.o
if errorlevel 1 goto error

"%GPP%" %CXXFLAGS% -c secp256k1/Point.cpp -o Point.o
if errorlevel 1 goto error

"%GPP%" %CXXFLAGS% -c secp256k1/SECP256K1.cpp -o SECP256K1.o
if errorlevel 1 goto error

"%GPP%" %CXXFLAGS% -c secp256k1/IntMod.cpp -o IntMod.o
if errorlevel 1 goto error

"%GPP%" %CXXFLAGS% -c secp256k1/Random.cpp -o Random.o
if errorlevel 1 goto error

"%GPP%" %CXXFLAGS% -c secp256k1/IntGroup.cpp -o IntGroup.o
if errorlevel 1 goto error

REM Compile hash modules
echo Compiling hash modules...
"%GPP%" %CXXFLAGS% -c hash/ripemd160.cpp -o hash/ripemd160.o
if errorlevel 1 goto error

"%GPP%" %CXXFLAGS% -c hash/sha256.cpp -o hash/sha256.o
if errorlevel 1 goto error

"%GPP%" %CXXFLAGS% -c hash/ripemd160_sse.cpp -o hash/ripemd160_sse.o
if errorlevel 1 goto error

"%GPP%" %CXXFLAGS% -c hash/sha256_sse.cpp -o hash/sha256_sse.o
if errorlevel 1 goto error

REM GPU backend
echo Compiling GPU backend...
"%GPP%" %CXXFLAGS% -c gpu_backend.cpp -o gpu_backend.o
if errorlevel 1 goto error

echo Linking final executable...
REM Link with GPU backend and KEYHUNT-ECC library
"%GPP%" %CXXFLAGS% -o keyhunt.exe keyhunt.cpp base58.o rmd160.o hash/ripemd160.o hash/ripemd160_sse.o hash/sha256.o hash/sha256_sse.o bloom.o oldbloom.o xxhash.o util.o Int.o Point.o SECP256K1.o IntMod.o Random.o IntGroup.o sha3.o keccak.o gpu_backend.o -L"%KEYHUNT_ECC_LIBDIR%" -lkeyhunt_ecc -L"%CUDA_LIBDIR%" -lcudart -lm -lpthread
if errorlevel 1 goto error

echo Cleaning up object files...
del *.o 2>nul
del hash\*.o 2>nul

echo Build completed successfully!
echo keyhunt.exe generated.
if exist keyhunt.exe (
    echo Executable size: 
    for %%i in (keyhunt.exe) do echo %%~zi bytes
)
goto end

:error
echo Build failed!
pause
exit /b 1

:end
echo Done.