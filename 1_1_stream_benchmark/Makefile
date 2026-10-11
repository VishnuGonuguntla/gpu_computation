TEST_CASE = stream
BUILD_DIR = build
CXX ?= g++
NVCC ?= nvcc
CXXFLAGS ?= -O3 -march=native -std=c++20
NVCCFLAGS ?= -O3 -std=c++20 -arch=sm_80

NX    = 10000000
NWARM = 2
NIT   = 10
PARAMETERS = ${NX} ${NWARM} ${NIT}

targets = $(BUILD_DIR)/$(TEST_CASE)-base $(BUILD_DIR)/$(TEST_CASE)-omp-host $(BUILD_DIR)/$(TEST_CASE)-cuda

.PHONY: all mk-target-dir build-serial build-parallel bench clean
.PHONY: $(TEST_CASE)-base $(TEST_CASE)-omp-host $(TEST_CASE)-cuda

all: mk-target-dir $(targets)

mk-target-dir:
	mkdir -p $(BUILD_DIR)

build-serial: mk-target-dir $(BUILD_DIR)/$(TEST_CASE)-base

build-parallel: mk-target-dir $(BUILD_DIR)/$(TEST_CASE)-omp-host $(BUILD_DIR)/$(TEST_CASE)-cuda

$(BUILD_DIR)/$(TEST_CASE)-base: $(TEST_CASE)-base.cpp $(TEST_CASE)-util.h util.h
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_CASE)-base.cpp

$(BUILD_DIR)/$(TEST_CASE)-omp-host: $(TEST_CASE)-omp-host.cpp $(TEST_CASE)-util.h util.h
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $(TEST_CASE)-omp-host.cpp

$(BUILD_DIR)/$(TEST_CASE)-cuda: $(TEST_CASE)-cuda.cu $(TEST_CASE)-util.h util.h ../cuda-util.h
	$(NVCC) $(NVCCFLAGS) -o $@ $(TEST_CASE)-cuda.cu

$(TEST_CASE)-base: $(BUILD_DIR)/$(TEST_CASE)-base
$(TEST_CASE)-omp-host: $(BUILD_DIR)/$(TEST_CASE)-omp-host
$(TEST_CASE)-cuda: $(BUILD_DIR)/$(TEST_CASE)-cuda

bench: all
	@echo "Base:"; $(BUILD_DIR)/$(TEST_CASE)-base $(PARAMETERS)
	@echo ""; echo "OpenMP Host:"; $(BUILD_DIR)/$(TEST_CASE)-omp-host $(PARAMETERS)
	@echo ""; echo "CUDA:"; $(BUILD_DIR)/$(TEST_CASE)-cuda $(PARAMETERS)

clean:
	rm -f $(targets)
