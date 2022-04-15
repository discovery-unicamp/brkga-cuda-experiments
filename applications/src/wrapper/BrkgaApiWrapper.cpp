#include "BrkgaApiWrapper.hpp"

struct BrkgaApiWrapper::InstanceWrapper {};

struct BrkgaApiWrapper::BrkgaWrapper {
    BrkgaWrapper(const BrkgaConfiguration& config) : algorithm(config.chromosomeLength, config.populationSize, config.getEliteProbability(), config.getMutantsProbability(), config.rhoe, MTRand(config.seed), config.numberOfPopulations, OMP_THREADS?) {}
private:
    BRKGA<InstanceWrapper, MTRand> algorithm;
};

BrkgaApiWrapper::BrkgaApiWrapper(const BrkgaConfiguration& config) : instance(new InstanceWrapper(config)), brkga(new BrkgaWrapper(config)) {}
