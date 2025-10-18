# Top-level Makefile for KEYHUNT-ECC project
# Forward all make commands to albertobsd-keyhunt subdirectory

.PHONY: all clean legacy bsgsd help verify-stage1

# Default target: build keyhunt with GPU support
all:
	@echo "[+] Building KEYHUNT-ECC project..."
	@$(MAKE) -C albertobsd-keyhunt

# Clean build artifacts
clean:
	@$(MAKE) -C albertobsd-keyhunt clean

# Build legacy version (GMP-based, no GPU)
legacy:
	@$(MAKE) -C albertobsd-keyhunt legacy

# Build bsgsd tool
bsgsd:
	@$(MAKE) -C albertobsd-keyhunt bsgsd

# Stage 1 build verification helper
verify-stage1:
	@./scripts/verify_build_stage1.sh

# Show help information
help:
	@echo "KEYHUNT-ECC Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make          - Build keyhunt with GPU support (auto-builds KEYHUNT-ECC library)"
	@echo "  make clean    - Clean all build artifacts"
	@echo "  make legacy   - Build legacy version (GMP-based, no GPU)"
	@echo "  make bsgsd    - Build bsgsd tool"
	@echo "  make verify-stage1 - Run Stage 1 Make/CMake verification with logs"
	@echo ""
	@echo "The keyhunt binary will be in: albertobsd-keyhunt/keyhunt"
