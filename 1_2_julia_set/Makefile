# configuration

TEST_CASE = julia
BUILD_DIR = build
CC = g++
NVCC = nvcc

# default benchmark parameters

WIDTH    = 800
HEIGHT = 800
MAXITER   = 100

PARAMETERS = ${WIDTH} ${HEIGHT} ${MAXITER}


# all
targets = $(BUILD_DIR)/$(TEST_CASE)-base $(BUILD_DIR)/$(TEST_CASE)-cuda 


.PHONY: all
all: mk-target-dir $(targets)


mk-target-dir:
	@mkdir -p $(BUILD_DIR)


# build rules

$(BUILD_DIR)/$(TEST_CASE)-base: $(TEST_CASE)-base.cpp util.h
	${CC} -O3 -march=native -std=c++17 -o $(BUILD_DIR)/$(TEST_CASE)-base $(TEST_CASE)-base.cpp

$(BUILD_DIR)/$(TEST_CASE)-cuda: $(TEST_CASE)-cuda.cu util.h ../cuda-util.h
	${NVCC} -O3 -std=c++17 -o $(BUILD_DIR)/$(TEST_CASE)-cuda $(TEST_CASE)-cuda.cu

# aliases without build directory

.PHONY: $(TEST_CASE)-base
$(TEST_CASE)-base: $(BUILD_DIR)/$(TEST_CASE)-base

.PHONY: $(TEST_CASE)-cuda
$(TEST_CASE)-cuda: $(BUILD_DIR)/$(TEST_CASE)-cuda


# automated benchmark target

.PHONY: bench
bench: all
	@echo "Base:"
	@$(BUILD_DIR)/$(TEST_CASE)-base $(PARAMETERS)
	@echo ""

	@echo "CUDA:"
	@$(BUILD_DIR)/$(TEST_CASE)-cuda $(PARAMETERS)
	@echo ""


# clean target

.PHONY: clean
clean:
	rm $(targets)