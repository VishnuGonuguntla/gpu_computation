# configuration

TEST_CASE = stream
SRC_DIR = src
BUILD_DIR = build
CC = g++
NVCC = nvcc
TARGET = $(BUILD_DIR)/my_program

C_FLAGS = -O3 -march=native -std=c++17
CUDA_FLAGS = -O3 -std=c++17

CPP_SOURCES = $(shell find $(SRC_DIR) -type f -name '*.cpp')
CUDA_CPP_SOURCES = $(shell find $(SRC_DIR) -type f -name '*.cu')

CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CPP_SOURCES))
CUDA_CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CUDA_CPP_SOURCES))
OBJECTS = $(CPP_OBJECTS) $(CUDA_CPP_OBJECTS)

.PHONY: all clean

all: $(TARGET)

$(TARGET) : $(OBJECTS)
	mkdir -p $(BUILD_DIR)
	$(CC) $(C_FLAGS) $(OBJECTS) -o $(TARGET)

$(BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CC) $(C_FLAGS) -c $< -o $@

$(BUILD_DIR)/%.o : $(SRC_DIR)/%.cu
	mkdir -p $(dir $@)
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

$(BUILD_DIR)/%: $(SRC_DIR)/%.cpp $(OBJECTS)
	@mkdir -p $(@D)
	$(CC) $(C_FLAGS) $< $(OBJECTS) -o $@

.SECONDARY: $(OBJECTS) 

clean:
	rm -rf $(BUILD_DIR)


# define compile_and_run
# 	@echo "=== Building $(1) ==="
# 	@mkdir -p $(BIN_DIR)
# 	$(CC) $(CFLAGS) $(SRC_DIR)/$(1)/*.c -o $(BIN_DIR)/$(1)
# 	@echo "=== Running $(1) ==="
# 	./$(BIN_DIR)/$(1)
# 	@echo "====================\n"
# endef



# targets = $(BUILD_DIR)/$(TEST_CASE)-base $(BUILD_DIR)/$(TEST_CASE)-omp-host


# .PHONY: all
# all: mk-target-dir $(targets)


# mk-target-dir:
# 	mkdir -p $(BUILD_DIR)


# # build rules

# $(BUILD_DIR)/$(TEST_CASE)-base: $(TEST_CASE)-base.cpp $(TEST_CASE)-util.h util.h
# 	g++ -O3 -march=native -std=c++17 -o $(BUILD_DIR)/$(TEST_CASE)-base $(TEST_CASE)-base.cpp

# $(BUILD_DIR)/$(TEST_CASE)-omp-host: $(TEST_CASE)-omp-host.cpp $(TEST_CASE)-util.h util.h
# 	g++ -O3 -march=native -std=c++17 -fopenmp -o $(BUILD_DIR)/$(TEST_CASE)-omp-host $(TEST_CASE)-omp-host.cpp


# # aliases without build directory

# .PHONY: $(TEST_CASE)-base
# $(TEST_CASE)-base: $(BUILD_DIR)/$(TEST_CASE)-base

# .PHONY: $(TEST_CASE)-omp-host
# $(TEST_CASE)-omp-host: $(BUILD_DIR)/$(TEST_CASE)-omp-host


# # automated benchmark target

# .PHONY: bench
# bench: all
# 	@echo "Base:"
# 	$(BUILD_DIR)/$(TEST_CASE)-base $(PARAMETERS)
# 	@echo ""
# 	@echo "OpenMP Host:"
# 	$(BUILD_DIR)/$(TEST_CASE)-omp-host $(PARAMETERS)
# 	@echo ""


# # clean target

# .PHONY: clean
# clean:
# 	rm $(targets)
