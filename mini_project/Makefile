BUILD_DIR = build
SRC_DIR = src

NVCC ?= nvcc
CXX ?= g++
NVCCFLAGS ?= -O3 -std=c++14 -arch=sm_80
CXXFLAGS ?= -O3 -march=native -std=c++17
LDFLAGS = -lcurand

SERIAL_SRC = $(SRC_DIR)/main-serial.cpp $(SRC_DIR)/Car.cpp $(SRC_DIR)/Track.cpp $(SRC_DIR)/MPPI.cpp $(SRC_DIR)/IOManager.cpp
CUDA_SRC = $(SRC_DIR)/main.cu $(SRC_DIR)/CudaMPPI.cu $(SRC_DIR)/Car.cpp $(SRC_DIR)/Track.cpp $(SRC_DIR)/IOManager.cpp

DATA_FILES = config.par carsConfig.par trackData.txt config_smoke.par
VIZ = $(SRC_DIR)/visualizer.py

targets = $(BUILD_DIR)/mppi-serial $(BUILD_DIR)/mppi-cuda

.PHONY: all serial cuda data clean

all: serial cuda data

serial: $(BUILD_DIR)/mppi-serial data

cuda: $(BUILD_DIR)/mppi-cuda data

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/mppi-serial: $(SERIAL_SRC) $(SRC_DIR)/types.h $(SRC_DIR)/dynamics.h $(SRC_DIR)/cost.h $(SRC_DIR)/Car.h $(SRC_DIR)/Track.h $(SRC_DIR)/MPPI.h $(SRC_DIR)/IOManager.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -o $@ $(SERIAL_SRC)

$(BUILD_DIR)/mppi-cuda: $(CUDA_SRC) $(SRC_DIR)/kernels.cuh $(SRC_DIR)/CudaMPPI.cuh $(SRC_DIR)/cuda-util.cuh $(SRC_DIR)/types.h $(SRC_DIR)/dynamics.h $(SRC_DIR)/cost.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(SRC_DIR) -o $@ $(CUDA_SRC) $(LDFLAGS)

data: | $(BUILD_DIR)
	cp -f $(DATA_FILES) $(BUILD_DIR)/
	cp -f $(VIZ) $(BUILD_DIR)/visualizer.py

clean:
	rm -f $(BUILD_DIR)/mppi-serial $(BUILD_DIR)/mppi-cuda \
	      $(BUILD_DIR)/telemetry.txt $(BUILD_DIR)/telemetry_serial.txt \
	      $(BUILD_DIR)/run_info.txt
