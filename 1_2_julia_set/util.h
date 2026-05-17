#pragma once

#include <cstdlib>
#include <iostream>
#include <chrono>

void parseCLA_1d(int argc, char *const *argv, size_t &WIDTH, size_t &HEIGHT, size_t &maxIter) {
    // default values
    WIDTH = 800;
    HEIGHT = 800;
    maxIter = 100;

    // override with command line arguments
    int i = 1;
    if (argc > i) WIDTH = atoi(argv[i]);
    ++i;
    if (argc > i) HEIGHT = atoi(argv[i]);
    ++i;
    if (argc > i) maxIter = atoi(argv[i]);
    ++i;
}


void printStats(const std::chrono::duration<double> elapsedSeconds, size_t particles, size_t iterations) {
    std::cout << "  #Particles / #Iterations:  " << particles << " / " << iterations << "\n";
    std::cout << "  elapsed time:  " << elapsedSeconds.count() << " s\n";
    std::cout << "  MLUP/s:        " << 1e-6 * particles * iterations / elapsedSeconds.count() << "\n";
}