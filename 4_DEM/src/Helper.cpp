#include <iostream>
#include <map>
#include <string>
#include <fstream>
#include <chrono>

inline void initialStats(std::map<std::string, double>& parameters) {
    std::cout << "Particles: " << parameters["nParticles"] << std::endl;
    std::cout << "StepSize: " << parameters["timeStep"] << std::endl;
    std::cout << "TotalTime: " << parameters["nTime"] <<  " Iterations: " <<  parameters["nTime"] / parameters["timeStep"]<< std::endl;
}

inline void parseParameter(std::string filename, std::map<std::string, double>& parameters) {
    std::ifstream file;
    std::string line;
    file.open(filename);
    if (file.is_open()) {
        while (getline(file, line)) {
            if (line.empty() || line[0] == '#')
                continue;
            size_t pos = line.find_first_of(" ");
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                parameters[key] = std::stod(value);
            }
        }
        file.close();
    } else {
        std::cerr << "!!! Error Opening File" << std::endl;
        return;
    }
}

inline void printStats(const std::chrono::duration<double> elapsedSeconds, size_t particles, size_t iterations) {
    std::cout << "  #Particles / #Iterations:  " << particles << " / " << iterations << "\n";
    std::cout << "  elapsed time:  " << elapsedSeconds.count() << " s\n";
    std::cout << "  MLUP/s:        " << 1e-6 * particles * iterations / elapsedSeconds.count() << "\n";
}