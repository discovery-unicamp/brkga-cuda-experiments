#include "GpuBrkgaWrapper.hpp"
#include <GPU-BRKGA/GPUBRKGA.cuh>
#include <brkga_cuda_api/BrkgaConfiguration.hpp>
#include <brkga_cuda_api/CudaUtils.hpp>
#include <brkga_cuda_api/Decoder.hpp>
#include <brkga_cuda_api/Logger.hpp>

#include <cstdlib>
#include <vector>

struct GpuBrkgaWrapper::DecoderWrapper {
public:
  DecoderWrapper(const BrkgaConfiguration& config)
      : decoder(config.decoder),
        chromosomeCount(config.populationSize),
        chromosomeLength(config.chromosomeLength),
        hostDecode(config.decodeType == DecodeType::HOST) {}

  void Init() const {}

  void Decode(float* chromosomes, float* fitness) const {
    CUDA_CHECK_LAST();  // Check for errors in GPU-BRKGA
    if (hostDecode) {
      decoder->hostDecode(chromosomeCount, chromosomes, fitness);
    } else {
      // The GPU-BRKGA doesn't provide any stream; we use the default instead
      decoder->deviceDecode(nullptr, chromosomeCount, chromosomes, fitness);
    }
  }

  Decoder* decoder;
  unsigned chromosomeCount;
  unsigned chromosomeLength;
  bool hostDecode;
};

struct GpuBrkgaWrapper::BrkgaWrapper {
  BrkgaWrapper(const BrkgaConfiguration& config, DecoderWrapper* decoder)
      : algorithm(config.chromosomeLength,
                  config.populationSize,
                  (double)config.eliteCount / (double)config.populationSize,
                  (double)config.mutantsCount / (double)config.populationSize,
                  config.rhoe,
                  *decoder,
                  config.seed,
                  /* decode on gpu? */ !decoder->hostDecode,
                  config.numberOfPopulations) {
    if (config.chromosomeLength > max_t) {
      logger::error("The chromosome length exceeds the thread limit:",
                    config.chromosomeLength, ">", max_t);
      abort();
    }
    if (config.decodeType != DecodeType::DEVICE
        && config.decodeType != DecodeType::HOST) {
      logger::error("Decode type", toString(config.decodeType),
                    "isn't supported; use either DEVICE or HOST");
      abort();
    }
  }

  GPUBRKGA<DecoderWrapper> algorithm;
};

GpuBrkgaWrapper::GpuBrkgaWrapper(const BrkgaConfiguration& config)
    : decoder(new DecoderWrapper(config)),
      brkga(new BrkgaWrapper(config, decoder)) {}

GpuBrkgaWrapper::~GpuBrkgaWrapper() {
  delete brkga;
  delete decoder;
}

void GpuBrkgaWrapper::evolve() {
  brkga->algorithm.evolve();
  CUDA_CHECK_LAST();
}

void GpuBrkgaWrapper::exchangeElite(unsigned count) {
  brkga->algorithm.exchangeElite(count);
  CUDA_CHECK_LAST();
}

float GpuBrkgaWrapper::getBestFitness() {
  auto best = brkga->algorithm.getBestIndividual();
  CUDA_CHECK_LAST();
  return best.fitness.first;
}

std::vector<float> GpuBrkgaWrapper::getBestChromosome() {
  auto best = brkga->algorithm.getBestIndividual();
  CUDA_CHECK_LAST();
  return std::vector<float>(best.aleles,
                            best.aleles + decoder->chromosomeLength);
}
