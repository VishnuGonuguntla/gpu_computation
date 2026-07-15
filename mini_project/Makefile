# configuration
BUILD_DIR = build
SRC_DIR = src
TEST_CASE = mppi
# target executable
targets = mppi-cuda

# compiler and flags
NVCC = nvcc
GCC = gcc
# Added -g -G for debugging, removed -O3 because optimization 
# interferes with accurate line-by-line stepping
NVCCFLAGS = -std=c++14 -arch=sm_80
#NVCCFLAGS = -O3 -std=c++14 
LDFLAGS = -lcurand

.PHONY: all
all: mk-target-dir $(targets)

mk-target-dir:
	mkdir -p $(BUILD_DIR)

# build rules
$(targets): $(SRC_DIR)/main.cu $(SRC_DIR)/Track.cpp $(SRC_DIR)/Car.cpp $(SRC_DIR)/Helper.cpp $(SRC_DIR)/CudaMPPI.cu $(SRC_DIR)/IOManager.cpp $(SRC_DIR)/kernels.cuh 
	${NVCC} $(NVCCFLAGS) -o $(TEST_CASE)-cuda $(SRC_DIR)/main.cu $(SRC_DIR)/Helper.cpp $(SRC_DIR)/Track.cpp $(SRC_DIR)/Car.cpp $(SRC_DIR)/CudaMPPI.cu $(SRC_DIR)/IOManager.cpp 
# aliases without build directory
.PHONY: mppi 
mppi: mppi-cuda

# clean target
.PHONY: clean sanitize-clean sanitize-memory
clean:
	rm -f $(targets)
	rm -r obj/*

sanitize-race: all
	compute-sanitizer --tool racecheck $(TEST_CASE)-cuda $(PARAMETERS)
sanitize-memory: all
	compute-sanitizer --tool memcheck $(TEST_CASE)-cuda $(PARAMETERS)