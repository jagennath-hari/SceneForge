#include "stereoforge/optimization/bundle_adjuster.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: stereoforge-bundle-adjust INPUT.json OUTPUT.json DEVICE\n";
            return 2;
        }
        const std::filesystem::path output(argv[2]);
        const std::filesystem::path temporary(output.string()+".partial");
        if (std::filesystem::exists(output) || std::filesystem::exists(temporary)) {
            throw std::runtime_error("Output already exists; use a fresh diagnostic directory");
        }
        std::ifstream input(argv[1]);
        input.exceptions(std::ios::badbit);
        if (!input) { throw std::runtime_error("Cannot open input JSON"); }
        nlohmann::json request;
        input >> request;
        const stereoforge::optimization::BundleAdjuster solver(std::stoi(argv[3]));
        const nlohmann::json result = solver.Solve(request);
        std::ofstream stream(temporary);
        stream.exceptions(std::ios::failbit | std::ios::badbit);
        stream << result.dump() << '\n';
        stream.close();
        std::filesystem::rename(temporary, output);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cuNLS BA failed: " << error.what() << '\n';
        return 1;
    }
}
