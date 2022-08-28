#include "../Tweaks.hpp"  // Must be generated
#include "../common/Parameters.hpp"
#include <brkga-cuda/Brkga.hpp>
#include <brkga-cuda/BrkgaConfiguration.hpp>
#include <brkga-cuda/CudaError.cuh>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#if defined(TSP)
#include "../common/instances/TspInstance.hpp"
#include "decoders/TspDecoder.hpp"
typedef TspInstance Instance;
typedef TspDecoder DecoderImpl;
#elif defined(SCP)
#include "../common/instances/ScpInstance.hpp"
#include "decoders/ScpDecoder.hpp"
typedef ScpInstance Instance;
typedef ScpDecoder DecoderImpl;
#elif defined(CVRP) || defined(CVRP_GREEDY)
#include "../common/instances/CvrpInstance.hpp"
#include "decoders/CvrpDecoder.hpp"
typedef CvrpInstance Instance;
typedef CvrpDecoder DecoderImpl;
#else
#error No problem/instance/decoder defined
#endif  // Problem/Instance

namespace std {
template <class T1, class T2>
inline ostream& operator<<(ostream& out, const pair<T1, T2>& p) {
  return out << '(' << p.first << ',' << p.second << ')';
}

template <class T>
inline ostream& operator<<(ostream& out, const vector<T>& v) {
  bool flag = false;
  out << '[';
  for (const auto& x : v) {
    if (flag) {
      out << ',';
    } else {
      flag = true;
    }
    out << x;
  }
  return out << ']';
}
}  // namespace std

std::string decodeType;

inline bool contains(const std::string& str, const std::string& pattern) {
  return str.find(pattern) != std::string::npos;
}

__global__ void callThrutSort(float* dChromosome,
                              unsigned* dPermutation,
                              unsigned chromosomeLength) {
  thrust::device_ptr<float> keys(dChromosome);
  thrust::device_ptr<unsigned> vals(dPermutation);
  thrust::sort_by_key(thrust::device, keys, keys + chromosomeLength, vals);
}

void sortChromosomeToValidate(const float* chromosome,
                              unsigned* permutation,
                              unsigned size) {
  std::iota(permutation, permutation + size, 0);
  box::logger::info("Sorting the chromosome to validate according to",
                    "decode type:", decodeType);

  if (contains(decodeType, "permutation") || contains(decodeType, "gpu")) {
    auto* dChromosome = box::cuda::alloc<float>(nullptr, size);
    auto* dPermutation = box::cuda::alloc<unsigned>(nullptr, size);

    box::cuda::copy2d(nullptr, dChromosome, chromosome, size);
    box::cuda::copy2d(nullptr, dPermutation, permutation, size);

    if (contains(decodeType, "permutation")) {
      box::logger::debug("Using bb-segsort");
      box::cuda::segSort(nullptr, dChromosome, dPermutation, 1, size);
    } else {
      box::logger::debug("Using thrust::sort");
      callThrutSort<<<1, 1>>>(dChromosome, dPermutation, size);
    }
    box::cuda::sync();

    box::cuda::copy2h(nullptr, permutation, dPermutation, size);

    box::cuda::free(nullptr, dChromosome);
    box::cuda::free(nullptr, dPermutation);
  } else if (contains(decodeType, "cpu")) {
    box::logger::debug("Using std::sort");
    std::sort(permutation, permutation + size, [&](unsigned a, unsigned b) {
      return chromosome[a] < chromosome[b];
    });
  } else {
    box::logger::error("Unknown sort method for the decoder:", decodeType,
                       "\n\t=> on", __PRETTY_FUNCTION__);
    abort();
  }
}

void sortChromosomeToValidate(const double*, unsigned*, unsigned) {
  box::logger::error(__PRETTY_FUNCTION__, "should not be called");
  abort();
}

int main(int argc, char** argv) {
  box::logger::info("Parsing parameters");
  auto params = Parameters::parse(argc, argv);
  decodeType = params.decoder;

  box::logger::info("Reading instance");
  Instance instance = Instance::fromFile(params.instanceFileName);

  box::logger::info("Building the decoder");
  DecoderImpl decoder(&instance);

  box::logger::info("Building the configuration");
  auto config = box::BrkgaConfiguration::Builder()
                    .generations(params.generations)
                    .numberOfPopulations(params.numberOfPopulations)
                    .populationSize(params.populationSize)
                    .chromosomeLength(instance.chromosomeLength())
                    .eliteCount(params.getNumberOfElites())
                    .mutantsCount(params.getNumberOfMutants())
                    .rhoe(params.rhoe)
                    .exchangeBestInterval(params.exchangeBestInterval)
                    .exchangeBestCount(params.exchangeBestCount)
                    .seed(params.seed)
                    .decoder(&decoder)
                    .decodeType(box::DecodeType::fromString(params.decoder))
                    .threadsPerBlock(params.threadsPerBlock)
                    .ompThreads(params.ompThreads)
                    .build();

  box::cuda::Timer timer;

  box::logger::info("Building the algorithm");
  box::Brkga brkga(config);

  box::logger::info("Optimizing");
  std::vector<std::pair<float, unsigned>> convergence;
  convergence.push_back(
      {brkga.getBestFitness(), (unsigned)(timer.milliseconds() / 1000)});

  const auto totalTestTime = 10 * 60 * 1000;  // 10 minutes of experiment
  for (unsigned gen = 1; timer.milliseconds() < totalTestTime; ++gen) {
    brkga.evolve();
    if (gen % params.exchangeBestInterval == 0 && gen != params.generations)
      brkga.exchangeElite(params.exchangeBestCount);

    float fitness = brkga.getBestFitness();
    auto previousSecond = convergence.back().second;
    auto currentSecond = (unsigned)(timer.milliseconds() / 1000);
    if (currentSecond != previousSecond)
      convergence.push_back({fitness, currentSecond});
  }

  auto bestFitness = brkga.getBestFitness();
  auto bestChromosome = brkga.getBestChromosome();
  float elapsedMs = timer.milliseconds();

  box::logger::info("Optimization has finished after", elapsedMs / 1000,
                    "seconds with fitness:", bestFitness);

  std::cout << std::fixed << std::setprecision(6) << "ans=" << bestFitness
            << " elapsed=" << elapsedMs / 1000
            << " convergence=" << convergence << '\n';

  box::logger::info("Validating the solution");
  instance.validate(bestChromosome.data(), bestFitness);

  box::logger::info("Everything looks good!");
  box::logger::info("Exiting");
  return 0;
}
