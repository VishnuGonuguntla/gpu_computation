TEST_CASE = julia
BUILD_DIR = build
CXX ?= g++
NVCC ?= nvcc
CXXFLAGS ?= -O3 -march=native -std=c++20
NVCCFLAGS ?= -O3 -std=c++20 -arch=sm_80

WIDTH = 1000
HEIGHT = 1000
MAXITER = 100
C1 = -0.7269
C2 = 0.1889
PARAMETERS = ${WIDTH} ${HEIGHT} ${MAXITER} ${C1} ${C2}

targets = $(BUILD_DIR)/$(TEST_CASE)-base $(BUILD_DIR)/$(TEST_CASE)-cuda

.PHONY: all mk-target-dir build-serial build-parallel bench clean
.PHONY: $(TEST_CASE)-base $(TEST_CASE)-cuda

all: mk-target-dir $(targets)

mk-target-dir:
	mkdir -p $(BUILD_DIR) results

build-serial: mk-target-dir $(BUILD_DIR)/$(TEST_CASE)-base

build-parallel: mk-target-dir $(BUILD_DIR)/$(TEST_CASE)-cuda

$(BUILD_DIR)/$(TEST_CASE)-base: src/julia-base.cpp util.h include/stb_image_write.h
	$(CXX) $(CXXFLAGS) -I. -Isrc -o $@ src/julia-base.cpp

$(BUILD_DIR)/$(TEST_CASE)-cuda: src/julia-cuda.cu util.h include/stb_image_write.h ../cuda-util.h
	$(NVCC) $(NVCCFLAGS) -I. -Isrc -o $@ src/julia-cuda.cu

$(TEST_CASE)-base: $(BUILD_DIR)/$(TEST_CASE)-base
$(TEST_CASE)-cuda: $(BUILD_DIR)/$(TEST_CASE)-cuda

bench: all
	@echo "Base:"; $(BUILD_DIR)/$(TEST_CASE)-base $(PARAMETERS)
	@echo ""; echo "CUDA:"; $(BUILD_DIR)/$(TEST_CASE)-cuda $(PARAMETERS)

clean:
	rm -f $(targets)
