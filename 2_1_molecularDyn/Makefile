CXX     = g++
NVCC    = nvcc
CXXFLAGS= -g -O2
NVCCFLAGS= -g -O2

# object files
OBJ = src/Solver.o src/Solver_cuda.o

bin/molDyn: src/molDyn.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# compile pure C++ with g++
src/Solver.o: src/Solver.cpp src/Solver.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $

# compile CUDA with nvcc
src/Solver_cuda.o: src/cudaSolver.cu src/cudaSolver.cuh src/cudaSolver.hpp
	$(NVCC) $(NVCCFLAGS) -c -o $@ $

clean:
	rm -f bin/molDyn src/*.o