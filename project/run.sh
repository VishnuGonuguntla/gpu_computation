#!/bin/bash -l

source /etc/profile

module load  nvhpc

nvcc -g -c -DDEBUG src/main.cu -o obj/main.o
g++ -std=c++17 -Wall -Wextra -O3 -Isrc -c src/Helper.cpp -o obj/Helper.o

nvcc -g -lcurand -arch=sm_80 obj/Helper.o obj/main.o -o build/main -lm

./build/main