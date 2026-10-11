TEST_CASE = molDyn
BUILD_DIR = build
SRC_DIR = src
CXX ?= g++
NVCC ?= nvcc

C_FLAGS ?= -O3 -march=native -std=c++20
CUDA_FLAGS ?= -O3 -std=c++20 -arch=sm_80
PARAMETERS = molDyn.par 1

targets = $(BUILD_DIR)/$(TEST_CASE)-base $(BUILD_DIR)/$(TEST_CASE)-cuda

.PHONY: all mk-target-dir build-serial build-parallel bench serial parallel sanitize-race sanitize-memory clean

all: mk-target-dir $(targets)

mk-target-dir:
	mkdir -p $(BUILD_DIR) data

build-serial: mk-target-dir $(BUILD_DIR)/$(TEST_CASE)-base

build-parallel: mk-target-dir $(BUILD_DIR)/$(TEST_CASE)-cuda

$(BUILD_DIR)/$(TEST_CASE)-base: $(SRC_DIR)/$(TEST_CASE)-base.cpp $(SRC_DIR)/Helper.cpp $(SRC_DIR)/Solver.cpp $(SRC_DIR)/Solver.hpp
	$(CXX) $(C_FLAGS) -I$(SRC_DIR) -o $@ $(SRC_DIR)/$(TEST_CASE)-base.cpp $(SRC_DIR)/Solver.cpp

$(BUILD_DIR)/$(TEST_CASE)-cuda: $(SRC_DIR)/$(TEST_CASE)-cuda.cu $(SRC_DIR)/cudaSolver.cu $(SRC_DIR)/kernels.cuh ../cuda-util.cuh $(SRC_DIR)/cudaSolver.cuh
	$(NVCC) $(CUDA_FLAGS) -I$(SRC_DIR) -o $@ $(SRC_DIR)/$(TEST_CASE)-cuda.cu $(SRC_DIR)/cudaSolver.cu -lcurand

bench: all
	@echo "Base:"; $(BUILD_DIR)/$(TEST_CASE)-base $(PARAMETERS)
	@echo ""; echo "CUDA:"; $(BUILD_DIR)/$(TEST_CASE)-cuda $(PARAMETERS)

serial: build-serial
	$(BUILD_DIR)/$(TEST_CASE)-base $(PARAMETERS)

parallel: build-parallel
	$(BUILD_DIR)/$(TEST_CASE)-cuda $(PARAMETERS)

sanitize-race: build-parallel
	compute-sanitizer --tool racecheck $(BUILD_DIR)/$(TEST_CASE)-cuda $(PARAMETERS)

sanitize-memory: build-parallel
	compute-sanitizer --tool memcheck $(BUILD_DIR)/$(TEST_CASE)-cuda $(PARAMETERS)

clean:
	rm -f $(targets)
