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


void printStats(const std::chrono::duration<double> elapsedSeconds, size_t nCells, size_t maxIter) {
    std::cout << "  #cells / #maxIter:  " << nCells << " / " << maxIter << "\n";
    std::cout << "  elapsed time:  " << 1e3 * elapsedSeconds.count() << " ms\n";
    std::cout << "  MLUP/s:        " << 1e-6 * nCells * maxIter / elapsedSeconds.count() << "\n";
}