# Top-level Makefile for KEYHUNT-ECC project
# Forward all make commands to albertobsd-keyhunt subdirectory

.PHONY: all clean legacy bsgsd help

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

# Show help information
help:
	@echo "KEYHUNT-ECC Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make          - Build keyhunt with GPU support (auto-builds KEYHUNT-ECC library)"
	@echo "  make clean    - Clean all build artifacts"
	@echo "  make legacy   - Build legacy version (GMP-based, no GPU)"
	@echo "  make bsgsd    - Build bsgsd tool"
	@echo ""
	@echo "The keyhunt binary will be in: albertobsd-keyhunt/keyhunt"
