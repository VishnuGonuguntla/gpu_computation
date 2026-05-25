#!/bin/bash -l
source /etc/profile
module load nvhpc

nvcc -g -c src/molDyn-cuda.cu -o obj/molDyn.o
nvcc -g -lcurand ${NVCC_FLAGS} -c src/cudaSolver.cu -arch=sm_80 -o obj/cudaSolver.o
g++ -std=c++2a -Wall -Wextra -O3 -Isrc -c src/Helper.cpp -o obj/Helper.o

nvcc -g -lcurand -arch=sm_80 obj/molDyn.o obj/Helper.o obj/cudaSolver.o -o bin/molDyn-cuda -lm

./bin/molDyn-cuda molDyn.par
# compute-sanitizer --tool racecheck \
#                   ./bin/molDyn-cuda molDyn.par