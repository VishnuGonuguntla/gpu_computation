# configuration

TEST_CASE = julia
BUILD_DIR = ../build/$(TEST_CASE)



# all - so this like what you wat your executable to be called as

targets = $(BUILD_DIR)/$(TEST_CASE)-serial $(BUILD_DIR)/$(TEST_CASE)-cuda 


.PHONY: all
all: mk-target-dir $(targets)

mk-target-dir:
	mkdir -p $(BUILD_DIR)


# build rules - something you are copying he thing on to thte terminal thing 

$(BUILD_DIR)/$(TEST_CASE)-serial: $(TEST_CASE)-serial.cpp lodepng.h lodepng.cpp # pre - requestites
	g++ -O3 -march=native -std=c++17 -o $(BUILD_DIR)/$(TEST_CASE)-serial $(TEST_CASE)-serial.cpp lodepng.cpp 

$(BUILD_DIR)/$(TEST_CASE)-cuda: $(TEST_CASE)-cuda.cu lodepng.h lodepng.cpp # pre - requestites
	nvcc -O3 -std=c++17 -o $(BUILD_DIR)/$(TEST_CASE)-cuda $(TEST_CASE)-cuda.cu lodepng.cpp 

# aliases without build directory

.PHONY: $(TEST_CASE)-serial 
$(TEST_CASE)-serial: $(BUILD_DIR)/$(TEST_CASE)-serial

.PHONY: $(TEST_CASE)-cuda 
$(TEST_CASE)-cuda: $(BUILD_DIR)/$(TEST_CASE)-cuda


# clean target

.PHONY: clean
clean:
	rm $(targets)
