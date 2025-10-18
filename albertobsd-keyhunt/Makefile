# GPU integration variables (customize if needed)

CXX ?= g++
CC  ?= gcc

COMMON_ARCH_FLAGS := -m64 -march=native -mtune=native -mssse3
CXX_BASE_FLAGS    := $(COMMON_ARCH_FLAGS) -Wall -Wextra -Wno-deprecated-copy -Ofast -ftree-vectorize
C_BASE_FLAGS      := $(COMMON_ARCH_FLAGS) -Wall -Wextra -Ofast -ftree-vectorize

CPPFLAGS ?=
CFLAGS   ?=
CXXFLAGS ?=
CUDAFLAGS ?=
LDFLAGS  ?=

ifeq ($(OS),Windows_NT)
    # Windows 路径 (MinGW/MSYS2)
    KEYHUNT_ECC_LIBDIR ?= D:/mybitcoin/2/keyhunt/gECC-main/KEYHUNT-ECC/build
    CUDA_INCDIR        ?= "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/include"
    CUDA_LIBDIR        ?= "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/lib/x64"
else
    # Linux/WSL 路径
    KEYHUNT_ECC_LIBDIR ?= ../KEYHUNT-ECC/build
endif

# Helper to locate CUDA toolkit resources when paths are not provided
NVCC := $(shell command -v nvcc 2>/dev/null)

define find_dir_with_file
$(firstword $(foreach dir,$(1),$(if $(wildcard $(dir)/$(2)),$(dir))))
endef

CUDA_INC_CANDIDATES :=
CUDA_LIB_CANDIDATES :=

ifneq ($(strip $(CUDA_INCDIR)),)
    CUDA_INC_CANDIDATES += $(CUDA_INCDIR)
endif
ifneq ($(strip $(CUDA_LIBDIR)),)
    CUDA_LIB_CANDIDATES += $(CUDA_LIBDIR)
endif

ifneq ($(strip $(NVCC)),)
    NVCC_BIN_DIR   := $(dir $(NVCC))
    NVCC_TOOLKIT   := $(abspath $(NVCC_BIN_DIR)/..)
    CUDA_INC_CANDIDATES += $(NVCC_TOOLKIT)/include
    CUDA_LIB_CANDIDATES += $(NVCC_TOOLKIT)/lib64 $(NVCC_TOOLKIT)/lib
endif

CUDA_INC_CANDIDATES += /usr/local/cuda/include /usr/include
CUDA_LIB_CANDIDATES += /usr/local/cuda/lib64 /usr/local/cuda/lib64/stubs /usr/lib/x86_64-linux-gnu /usr/lib/wsl/lib

CUDA_INCDIR ?= $(call find_dir_with_file,$(CUDA_INC_CANDIDATES),cuda_runtime.h)
CUDA_LIBDIR ?= $(call find_dir_with_file,$(CUDA_LIB_CANDIDATES),libcudart.so)

ifeq ($(strip $(CUDA_INCDIR)),)
    $(error Unable to locate CUDA include directory. Please set CUDA_INCDIR.)
endif
ifeq ($(strip $(CUDA_LIBDIR)),)
    $(error Unable to locate CUDA library directory. Please set CUDA_LIBDIR.)
endif

CPPFLAGS += -I$(CUDA_INCDIR)

# KEYHUNT-ECC library path
KEYHUNT_ECC_LIB    := $(KEYHUNT_ECC_LIBDIR)/libkeyhunt_ecc.a
KEYHUNT_ECC_SOURCE := $(abspath $(KEYHUNT_ECC_LIBDIR)/..)

# Optional CUDA architecture override (defaults to native detection in CMake)
CUDA_ARCH ?=

# Set default goal to build keyhunt (not the library)
.DEFAULT_GOAL := default

.PHONY: default clean legacy bsgsd

default: keyhunt

# Auto-build KEYHUNT-ECC library if not exists or outdated
$(KEYHUNT_ECC_LIB):
	@echo "[+] Building KEYHUNT-ECC GPU library..."
	@mkdir -p $(KEYHUNT_ECC_LIBDIR)
	@cd $(KEYHUNT_ECC_SOURCE) && \
		CXXFLAGS="$(CXXFLAGS)" CMAKE_CUDA_FLAGS="$(CUDAFLAGS)" cmake -B build \
			-DCMAKE_BUILD_TYPE=Release \
			$(if $(strip $(CUDA_ARCH)),-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH),) \
			-DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$(abspath $(KEYHUNT_ECC_LIBDIR))"
	@cmake --build $(KEYHUNT_ECC_LIBDIR) --config Release -- -j$$(nproc)
	@echo "[+] KEYHUNT-ECC library built successfully"

keyhunt: $(KEYHUNT_ECC_LIB)
	@echo "[+] Building keyhunt..."
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c oldbloom/bloom.cpp -o oldbloom.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c bloom/bloom.cpp -o bloom.o
	$(CC)  $(CPPFLAGS) $(C_BASE_FLAGS)   $(CFLAGS)   -Wno-unused-parameter -c base58/base58.c -o base58.o
	$(CC)  $(CPPFLAGS) $(C_BASE_FLAGS)   $(CFLAGS)   -c rmd160/rmd160.c -o rmd160.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c sha3/sha3.c -o sha3.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c sha3/keccak.c -o keccak.o
	$(CC)  $(CPPFLAGS) $(C_BASE_FLAGS)   $(CFLAGS)   -c xxhash/xxhash.c -o xxhash.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c util.c -o util.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c secp256k1/Int.cpp -o Int.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c secp256k1/Point.cpp -o Point.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c secp256k1/SECP256K1.cpp -o SECP256K1.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c secp256k1/IntMod.cpp -o IntMod.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c secp256k1/Random.cpp -o Random.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c secp256k1/IntGroup.cpp -o IntGroup.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c hash/ripemd160.cpp -o hash/ripemd160.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c hash/sha256.cpp -o hash/sha256.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c hash/ripemd160_sse.cpp -o hash/ripemd160_sse.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c hash/sha256_sse.cpp -o hash/sha256_sse.o
	# GPU backend
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c gpu_backend.cpp -o gpu_backend.o
	# Link with GPU backend and KEYHUNT-ECC library
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) \
		-flto -funroll-loops -finline-functions -fprefetch-loop-arrays \
		-pipe -fomit-frame-pointer \
		-o keyhunt keyhunt.cpp base58.o rmd160.o hash/ripemd160.o hash/ripemd160_sse.o \
		hash/sha256.o hash/sha256_sse.o bloom.o oldbloom.o xxhash.o util.o Int.o Point.o \
		SECP256K1.o IntMod.o Random.o IntGroup.o sha3.o keccak.o gpu_backend.o \
		-L$(KEYHUNT_ECC_LIBDIR) -lkeyhunt_ecc -L$(CUDA_LIBDIR) -lcudart -lm -lpthread \
		-Wl,--allow-multiple-definition $(LDFLAGS)
	@rm -f *.o hash/*.o

clean:
	rm -f keyhunt *.o hash/*.o

legacy:
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c oldbloom/bloom.cpp -o oldbloom.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c bloom/bloom.cpp -o bloom.o
	$(CC)  $(CPPFLAGS) $(C_BASE_FLAGS)   $(CFLAGS)   -Wno-unused-result -c base58/base58.c -o base58.o
	$(CC)  $(CPPFLAGS) $(C_BASE_FLAGS)   $(CFLAGS)   -c xxhash/xxhash.c -o xxhash.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c util.c -o util.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c sha3/sha3.c -o sha3.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c sha3/keccak.c -o keccak.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c hashing.c -o hashing.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c gmp256k1/Int.cpp -o Int.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c gmp256k1/Point.cpp -o Point.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c gmp256k1/GMP256K1.cpp -o GMP256K1.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c gmp256k1/IntMod.cpp -o IntMod.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c gmp256k1/Random.cpp -o Random.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c gmp256k1/IntGroup.cpp -o IntGroup.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) \
		-o keyhunt keyhunt_legacy.cpp base58.o bloom.o oldbloom.o xxhash.o util.o Int.o \
		Point.o GMP256K1.o IntMod.o IntGroup.o Random.o hashing.o sha3.o keccak.o -lm -lpthread -lcrypto -lgmp $(LDFLAGS)
	@rm -f *.o hash/*.o

bsgsd:
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c oldbloom/bloom.cpp -o oldbloom.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c bloom/bloom.cpp -o bloom.o
	$(CC)  $(CPPFLAGS) $(C_BASE_FLAGS)   $(CFLAGS)   -Wno-unused-parameter -c base58/base58.c -o base58.o
	$(CC)  $(CPPFLAGS) $(C_BASE_FLAGS)   $(CFLAGS)   -c rmd160/rmd160.c -o rmd160.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c sha3/sha3.c -o sha3.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c sha3/keccak.c -o keccak.o
	$(CC)  $(CPPFLAGS) $(C_BASE_FLAGS)   $(CFLAGS)   -c xxhash/xxhash.c -o xxhash.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c util.c -o util.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c secp256k1/Int.cpp -o Int.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c secp256k1/Point.cpp -o Point.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c secp256k1/SECP256K1.cpp -o SECP256K1.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c secp256k1/IntMod.cpp -o IntMod.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c secp256k1/Random.cpp -o Random.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c secp256k1/IntGroup.cpp -o IntGroup.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c hash/ripemd160.cpp -o hash/ripemd160.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c hash/sha256.cpp -o hash/sha256.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c hash/ripemd160_sse.cpp -o hash/ripemd160_sse.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) -c hash/sha256_sse.cpp -o hash/sha256_sse.o
	$(CXX) $(CPPFLAGS) $(CXX_BASE_FLAGS) $(CXXFLAGS) \
		-o bsgsd bsgsd.cpp base58.o rmd160.o hash/ripemd160.o hash/ripemd160_sse.o hash/sha256.o \
		hash/sha256_sse.o bloom.o oldbloom.o xxhash.o util.o Int.o Point.o SECP256K1.o IntMod.o \
		Random.o IntGroup.o sha3.o keccak.o -lm -lpthread $(LDFLAGS)
	@rm -f *.o hash/*.o
