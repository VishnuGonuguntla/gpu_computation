# configuration
BUILD_DIR = ../build

# target executable
targets = $(BUILD_DIR)/mppi

# compiler and flags
NVCC = nvcc
# Added -g -G for debugging, removed -O3 because optimization 
# interferes with accurate line-by-line stepping
NVCCFLAGS = -std=c++14 -g -G
#NVCCFLAGS = -O3 -std=c++14 
LDFLAGS = -lcurand

.PHONY: all
all: mk-target-dir $(targets)

mk-target-dir:
	mkdir -p $(BUILD_DIR)

# build rules
$(BUILD_DIR)/mppi: main.cpp Car.cpp CudaMPPI.cu IOManager.cpp Car.h CudaMPPI.cuh IOManager.h
	$(NVCC) $(NVCCFLAGS) -arch=sm_60 -o $(BUILD_DIR)/mppi main.cpp Car.cpp CudaMPPI.cu IOManager.cpp $(LDFLAGS)

# aliases without build directory
.PHONY: mppi 
mppi: $(BUILD_DIR)/mppi

# clean target
.PHONY: clean
clean:
	rm -f $(targets)