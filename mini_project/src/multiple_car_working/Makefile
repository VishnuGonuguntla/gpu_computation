# configuration

BUILD_DIR = ../build



# all - so this like what you wat your executable to be called as

targets = $(BUILD_DIR)/mppi

.PHONY: all
all: mk-target-dir $(targets)

mk-target-dir:
	mkdir -p $(BUILD_DIR)


# build rules - something you are copying he thing on to thte terminal thing 

$(BUILD_DIR)/mppi: main.cpp Car.cpp Track.cpp MPPI.cpp IOManager.cpp Car.h Track.h MPPI.h IOManager.h # pre - requestites
	g++ -O3 -march=native -std=c++17 -o $(BUILD_DIR)/mppi main.cpp Car.cpp Track.cpp MPPI.cpp IOManager.cpp


# aliases without build directory
.PHONY: mppi 
mppi: $(BUILD_DIR)/mppi

# clean target
.PHONY: clean
clean:
	rm -f $(targets)
