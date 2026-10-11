NVCC ?= nvcc
CXX ?= g++
NVCCFLAGS ?= -O3 -std=c++20 -arch=sm_80
CXXFLAGS ?= -O3 -march=native -std=c++20
LDFLAGS = -lcurand

SERIAL_SRC = $(SRC_DIR)/main-serial.cpp $(SRC_DIR)/Car.cpp $(SRC_DIR)/Track.cpp $(SRC_DIR)/MPPI.cpp $(SRC_DIR)/IOManager.cpp
CUDA_SRC = $(SRC_DIR)/main.cu $(SRC_DIR)/CudaMPPI.cu $(SRC_DIR)/Car.cpp $(SRC_DIR)/Track.cpp $(SRC_DIR)/IOManager.cpp

DATA_FILES = config.par carsConfig.par trackData.txt
VIZ = $(SRC_DIR)/visualizer.py

targets = mppi-serial mppi-cuda

.PHONY: all serial cuda data clean build-serial build-parallel

all: serial cuda data

build-serial: mppi-serial data

build-parallel: mppi-cuda data

serial: build-serial

cuda: build-parallel


mppi-serial: $(SERIAL_SRC) $(SRC_DIR)/types.h $(SRC_DIR)/dynamics.h $(SRC_DIR)/cost.h $(SRC_DIR)/Car.h $(SRC_DIR)/Track.h $(SRC_DIR)/MPPI.h $(SRC_DIR)/IOManager.h
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -o $@ $(SERIAL_SRC)

mppi-cuda: $(CUDA_SRC) $(SRC_DIR)/kernels.cuh $(SRC_DIR)/CudaMPPI.cuh $(SRC_DIR)/cuda-util.cuh $(SRC_DIR)/types.h $(SRC_DIR)/dynamics.h $(SRC_DIR)/cost.h
	$(NVCC) $(NVCCFLAGS) -I$(SRC_DIR) -o $@ $(CUDA_SRC) $(LDFLAGS)

clean:
	rm -f mppi-serial  mppi-cuda \
	      telemetry_serial.txt  telemetry_parallel.txt \
	      run_info.txt
