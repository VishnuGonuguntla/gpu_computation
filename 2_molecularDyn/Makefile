TEST_CASE = molDyn
BUILD_DIR = build
SRC_DIR = src
CC = g++
NVCC = nvcc

C_FLAGS = -O3 -march=native -std=c++17
CUDA_FLAGS = -g -lcurand -arch=sm_80 -O3 -std=c++17
PARAMETERS = molDyn.par 1

targets = $(BUILD_DIR)/$(TEST_CASE)-base $(BUILD_DIR)/$(TEST_CASE)-cuda

$(BUILD_DIR)/$(TEST_CASE)-base: $(SRC_DIR)/$(TEST_CASE)-base.cpp $(SRC_DIR)/Helper.cpp $(SRC_DIR)/Solver.cpp $(SRC_DIR)/util.h
	${CC} $(C_FLAGS) -o $(BUILD_DIR)/$(TEST_CASE)-base $(SRC_DIR)/$(TEST_CASE)-base.cpp $(SRC_DIR)/Solver.cpp

$(BUILD_DIR)/$(TEST_CASE)-cuda: $(SRC_DIR)/$(TEST_CASE)-cuda.cu $(SRC_DIR)/cudaSolver.cu $(SRC_DIR)/kernels.cuh $(SRC_DIR)/cuda-util.cuh
	${NVCC} $(CUDA_FLAGS) -o $(BUILD_DIR)/$(TEST_CASE)-cuda $(SRC_DIR)/$(TEST_CASE)-cuda.cu $(SRC_DIR)/cudaSolver.cu

.PHONY: all bench serial parallel sanitize-race sanitize-memory
all: mk-target-dir $(targets)

mk-target-dir:
	@mkdir -p $(BUILD_DIR)

bench: all
	@echo "Base:"
	$(BUILD_DIR)/$(TEST_CASE)-base $(PARAMETERS)
	@echo ""
	@echo "CUDA:"
	$(BUILD_DIR)/$(TEST_CASE)-cuda $(PARAMETERS)
	@echo ""

serial: all
	@echo "Base:"
	$(BUILD_DIR)/$(TEST_CASE)-base $(PARAMETERS)
parallel: all
	@echo "CUDA:"
	$(BUILD_DIR)/$(TEST_CASE)-cuda $(PARAMETERS)
	@echo ""
sanitize-race: all
	compute-sanitizer --tool racecheck $(BUILD_DIR)/$(TEST_CASE)-cuda $(PARAMETERS)
sanitize-memory: all
	compute-sanitizer --tool memcheck $(BUILD_DIR)/$(TEST_CASE)-cuda $(PARAMETERS)
.PHONY: clean
clean:
	rm $(targets)