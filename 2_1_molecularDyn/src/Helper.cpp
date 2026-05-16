#include <iostream>
#include <map>
#include <string>
#include <fstream>

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