#include "Parameters.hpp"

#include "Checker.hpp"
#include "Logger.hpp"

#include <iostream>

static const char* RESET = "\033[0m";
static const char* YELLOW = "\033[33m";

Parameters Parameters::parse(unsigned argc, char** argv) {
  box::logger::info("Reading parameters");

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

    box::logger::debug("Received:", arg, value);

    if (arg == "--instance") {
      params.instanceFileName = value;
    } else if (arg == "--generations") {
      params.generations = std::stoi(value);
    } else if (arg == "--max-time") {
      params.maxTimeSeconds = std::stof(value);
    } else if (arg == "--exchange-interval") {
      params.exchangeBestInterval = std::stoi(value);
    } else if (arg == "--exchange-count") {
      params.exchangeBestCount = std::stoi(value);
    } else if (arg == "--pr-interval") {
      params.prInterval = std::stoi(value);
    } else if (arg == "--pr-pairs") {
      params.prPairs = std::stoi(value);
    } else if (arg == "--pr-block-size") {
      params.prBlockSize = std::stoi(value);
    } else if (arg == "--pr-block-factor") {
      params.prBlockFactor = std::stof(value);
    } else if (arg == "--pr-min-diff") {
      params.prMinDiffPercentage = std::stof(value);
    } else if (arg == "--pr-max-time") {
      params.prMaxTime = std::stoi(value);
    } else if (arg == "--pr-select") {
      params.prSelect = value;
    } else if (arg == "--prune-interval") {
      params.pruneInterval = std::stoi(value);
    } else if (arg == "--prune-threshold") {
      params.pruneThreshold = std::stof(value);
    } else if (arg == "--pop-count") {
      params.numberOfPopulations = std::stoi(value);
    } else if (arg == "--pop-size") {
      params.populationSize = std::stoi(value);
    } else if (arg == "--nelite") {
      params.eliteSize = std::stoi(value);
    } else if (arg == "--elite") {
      params.elitePercentage = std::stof(value);
    } else if (arg == "--mutant") {
      params.mutantFactor = std::stof(value);
    } else if (arg == "--nmutant") {
      params.mutantSize = std::stoi(value);
    } else if (arg == "--rhoe") {
      params.rhoe = std::stof(value);
    } else if (arg == "--rhoe-function") {
      params.rhoeFunction = value;
    } else if (arg == "--parents") {
      params.numParents = std::stoi(value);
    } else if (arg == "--elite-parents") {
      params.numEliteParents = std::stoi(value);
    } else if (arg == "--seed") {
      params.seed = std::stoi(value);
    } else if (arg == "--decoder") {
      params.decoder = value;
    } else if (arg == "--threads") {
      params.threadsPerBlock = std::stoi(value);
    } else if (arg == "--omp-threads") {
      params.ompThreads = std::stoi(value);
    } else if (arg == "--log-step") {
      params.logStep = std::stoi(value);
    } else {
      std::cerr << YELLOW << "[WARNING] Unknown argument was ignored: " << arg
                << ' ' << value << RESET << '\n';
    }
  }

  return params;
}
