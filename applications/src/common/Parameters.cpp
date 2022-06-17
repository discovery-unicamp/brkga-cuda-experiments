#include "Parameters.hpp"

#include "Checker.hpp"

Parameters Parameters::parse(unsigned argc, char** argv) {
  Parameters params;
  for (unsigned i = 1; i < argc; i += 2) {
    std::string arg = argv[i];
    CHECK(arg.substr(0, 2) == "--",
          "All arguments should start with --; found %s", arg.c_str());
    CHECK(i + 1 < argc, "Missing value for %s", arg.c_str());

    std::string value = argv[i + 1];
    CHECK(value.substr(0, 2) != "--",
          "Argument value for %s starts with --: %s", arg.c_str(),
          value.c_str());

    if (arg == "--instance") {
      params.instanceFileName = value;
    } else if (arg == "--generations") {
      params.generations = std::stoi(value);
    } else if (arg == "--exchange-interval") {
      params.exchangeBestInterval = std::stoi(value);
    } else if (arg == "--exchange-count") {
      params.exchangeBestCount = std::stoi(value);
    } else if (arg == "--pop-count") {
      params.numberOfPopulations = std::stoi(value);
    } else if (arg == "--pop-size") {
      params.populationSize = std::stoi(value);
    } else if (arg == "--nelite") {
      params.eliteSize = std::stoi(value);
    } else if (arg == "--elite") {
      params.eliteProportion = std::stof(value);
    } else if (arg == "--mutant") {
      params.mutantProportion = std::stof(value);
    } else if (arg == "--nmutant") {
      params.mutantSize = std::stoi(value);
    } else if (arg == "--rhoe") {
      params.rhoe = std::stof(value);
    } else if (arg == "--seed") {
      params.seed = std::stoi(value);
    } else if (arg == "--decode") {
      params.decoder = value;
    } else if (arg == "--threads") {
      params.threadsPerBlock = std::stoi(value);
    } else if (arg == "--omp-threads") {
      params.ompThreads = std::stoi(value);
    } else if (arg == "--log-step") {
      params.logStep = std::stoi(value);
    } else {
      CHECK(false, "Unknown argument: %s", arg.c_str());
    }
  }

  return params;
}
