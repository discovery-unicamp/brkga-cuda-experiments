#include "aaa.cuh"

const unsigned THREADS_PER_BLOCK = 256;

__global__ void decodeImpl(float* dFitness,
                           float* dPopulation,
                           unsigned numPopulations,
                           unsigned populationSize,
                           unsigned chromosomeLength) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numPopulations * populationSize) return;

  auto* chromosome = dPopulation + tid * chromosomeLength;
  auto& fitness = dFitness[tid] = 0;
  for (unsigned i = 0; i < chromosomeLength; ++i)
    fitness += sinf(i + 1) * chromosome[i];
}

void decode(float* dFitness,
            unsigned* dFitnessIdx,
            float* dPopulation,
            unsigned numPopulations,
            unsigned populationSize,
            unsigned chromosomeLength) {
  unsigned threads = THREADS_PER_BLOCK;
  auto blocks = (numPopulations * populationSize + threads - 1) / threads;
  decodeImpl<<<blocks, threads>>>(dFitness, dPopulation, numPopulations,
                                  populationSize, chromosomeLength);

  gpu::iotaMod(nullptr, dFitnessIdx, numPopulations * populationSize,
               populationSize);
  for (unsigned p = 0; p < numPopulations; ++p) {
    auto offset = p * populationSize;
    gpu::sortByKey(nullptr, dFitness + offset, dFitnessIdx + offset,
                   populationSize);
  }
}

__global__ void deviceGetBest(float* dBest,
                              float* dFitness,
                              unsigned* dFitnessIdx,
                              unsigned numPopulations,
                              unsigned populationSize) {
  *dBest = INFINITY;
  for (unsigned p = 0; p < numPopulations; ++p) {
    unsigned k = dFitnessIdx[p * populationSize];
    float fitness = dFitness[k];
    if (fitness < *dBest) { *dBest = fitness; }
  }
}

float getBest(float* dFitness,
              unsigned* dFitnessIdx,
              unsigned numPopulations,
              unsigned populationSize) {
  auto* dBest = gpu::alloc<float>(nullptr, 1);
  deviceGetBest<<<1, 1>>>(dBest, dFitness, dFitnessIdx, numPopulations,
                          populationSize);
  float best = INFINITY;
  gpu::copy2h(nullptr, &best, dBest, 1);
  gpu::free(nullptr, dBest);
  gpu::sync();
  return best;
}

namespace v0 {  // brkga-cuda 1.0
__global__ void evolve(float* dNewPopulation,  // random rhoe is stored here
                       const float* dPopulation,
                       const unsigned* dFitnessIdx,
                       const float* dEliteParent,
                       const float* dParent,
                       const unsigned chromosomeLength,
                       const unsigned populationCount,
                       const unsigned populationSize,
                       const unsigned eliteSize,
                       const unsigned mutantsSize,
                       const float rhoe) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= populationCount * populationSize) return;

  auto* newChromosome = dNewPopulation + tid * chromosomeLength;
  auto populationIndex = tid / populationSize;
  auto chromosomeIndex = tid % populationSize;
  auto offset = populationIndex * populationSize;

  if (chromosomeIndex < eliteSize) {
    // survive: copy an elite
    auto* chromosome =
        dPopulation
        + (offset + dFitnessIdx[offset + chromosomeIndex]) * chromosomeLength;
    for (unsigned i = 0; i < chromosomeLength; ++i) {
      newChromosome[i] = chromosome[i];
    }
  } else if (chromosomeIndex < populationSize - mutantsSize) {
    // crossover: mate two chromosomes
    auto eliteParentIndex =
        (unsigned)(ceilf(dEliteParent[tid] * eliteSize) - 1);
    auto nonEliteParentIndex =
        (unsigned)(eliteSize
                   + ceilf(dParent[tid] * (populationSize - eliteSize)) - 1);

    auto* dPopFitnessIdx = dFitnessIdx + offset;
    auto* eliteParent =
        dPopulation
        + (offset + dPopFitnessIdx[eliteParentIndex]) * chromosomeLength;
    auto* nonEliteParent =
        dPopulation
        + (offset + dPopFitnessIdx[nonEliteParentIndex]) * chromosomeLength;

    for (unsigned i = 0; i < chromosomeLength; ++i) {
      newChromosome[i] =
          newChromosome[i] <= rhoe ? eliteParent[i] : nonEliteParent[i];
    }
  } else {
    // mutant: use the random values generated
  }
}
}  // namespace v0

namespace v1 {
__global__ void evolve(float* dNewPopulation,  // random rhoe
                       const float* dPopulation,
                       const unsigned* dFitnessIdx,
                       const float* dEliteParent,
                       const float* dParent,
                       const unsigned chromosomeLength,
                       const unsigned populationCount,
                       const unsigned populationSize,
                       const unsigned eliteSize,
                       const unsigned mutantsSize,
                       const float rhoe) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)populationCount * populationSize * chromosomeLength;
  if (tid >= total) return;

  auto geneIndex = tid % chromosomeLength;
  auto chromosomeIndex = tid / chromosomeLength % populationSize;
  auto populationIndex = tid / chromosomeLength / populationSize;
  auto offset = populationIndex * populationSize;
  auto* newChromosome =
      dNewPopulation + (offset + chromosomeIndex) * chromosomeLength;

  if (chromosomeIndex < eliteSize) {
    // survive: copy an elite
    auto* chromosome =
        dPopulation
        + (offset + dFitnessIdx[offset + chromosomeIndex]) * chromosomeLength;
    newChromosome[geneIndex] = chromosome[geneIndex];
  } else if (chromosomeIndex < populationSize - mutantsSize) {
    // crossover: mate two chromosomes
    auto eliteParentIndex =
        (unsigned)(ceilf(dEliteParent[tid / chromosomeLength] * eliteSize) - 1);
    auto nonEliteParentIndex =
        (unsigned)(eliteSize
                   + ceilf(dParent[tid / chromosomeLength]
                           * (populationSize - eliteSize))
                   - 1);

    auto* dPopFitnessIdx = dFitnessIdx + offset;
    auto* eliteParent =
        dPopulation
        + (offset + dPopFitnessIdx[eliteParentIndex]) * chromosomeLength;
    auto* nonEliteParent =
        dPopulation
        + (offset + dPopFitnessIdx[nonEliteParentIndex]) * chromosomeLength;

    newChromosome[geneIndex] = newChromosome[geneIndex] <= rhoe
                                   ? eliteParent[geneIndex]
                                   : nonEliteParent[geneIndex];
  } else {
    // mutant: use the random values generated
  }
}
}  // namespace v1

namespace v2 {
__global__ void evolve(float* dNewPopulation,  // random rhoe
                       const float* dPopulation,
                       const unsigned* dFitnessIdx,
                       const float* dEliteParent,
                       const float* dParent,
                       const unsigned chromosomeLength,
                       const unsigned populationCount,
                       const unsigned populationSize,
                       const unsigned eliteSize,
                       const unsigned mutantsSize,
                       const float rhoe) {
  __shared__ float sm[THREADS_PER_BLOCK];

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)populationCount * populationSize * chromosomeLength;
  if (tid >= total) return;

  auto geneIndex = tid % chromosomeLength;
  auto chromosomeIndex = tid / chromosomeLength % populationSize;
  auto populationIndex = tid / chromosomeLength / populationSize;
  auto offset = populationIndex * populationSize;
  auto* newChromosome =
      dNewPopulation + (offset + chromosomeIndex) * chromosomeLength;

  if (chromosomeIndex < eliteSize) {
    // survive: copy an elite
    auto* chromosome =
        dPopulation
        + (offset + dFitnessIdx[offset + chromosomeIndex]) * chromosomeLength;
    sm[threadIdx.x] = chromosome[geneIndex];
  } else if (chromosomeIndex < populationSize - mutantsSize) {
    // crossover: mate two chromosomes
    auto pid = tid / chromosomeLength;
    auto eliteParentIndex =
        (unsigned)(ceilf(dEliteParent[pid] * eliteSize) - 1);
    auto nonEliteParentIndex =
        (unsigned)(eliteSize
                   + ceilf(dParent[pid] * (populationSize - eliteSize)) - 1);

    auto* dPopFitnessIdx = dFitnessIdx + offset;
    auto* eliteParent =
        dPopulation
        + (offset + dPopFitnessIdx[eliteParentIndex]) * chromosomeLength;
    auto* nonEliteParent =
        dPopulation
        + (offset + dPopFitnessIdx[nonEliteParentIndex]) * chromosomeLength;

    sm[threadIdx.x] = newChromosome[geneIndex];
    sm[threadIdx.x] = sm[threadIdx.x] <= rhoe ? eliteParent[geneIndex]
                                              : nonEliteParent[geneIndex];
  } else {
    // mutant: use the random values generated
  }
  if (chromosomeIndex < populationSize - mutantsSize) {
    __syncthreads();
    newChromosome[geneIndex] = sm[threadIdx.x];
  }
}
}  // namespace v2

namespace v3 {
__global__ void evolveSurvive(float* dNewPopulation,
                              const float* dPopulation,
                              const unsigned* dFitnessIdx,
                              const unsigned chromosomeLength,
                              const unsigned populationCount,
                              const unsigned populationSize,
                              const unsigned eliteSize) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)populationCount * eliteSize * chromosomeLength;
  if (tid >= total) return;

  auto geneIndex = tid % chromosomeLength;
  auto chromosomeIndex = tid / chromosomeLength % eliteSize;
  auto populationIndex = tid / chromosomeLength / eliteSize;
  auto offset = populationIndex * populationSize;
  auto* newChromosome =
      dNewPopulation + (offset + chromosomeIndex) * chromosomeLength;

  auto* chromosome =
      dPopulation
      + (offset + dFitnessIdx[offset + chromosomeIndex]) * chromosomeLength;
  newChromosome[geneIndex] = chromosome[geneIndex];
}

__global__ void evolveCrossover(float* dNewPopulation,  // random rhoe
                                const float* dPopulation,
                                const unsigned* dFitnessIdx,
                                const float* dEliteParent,
                                const float* dParent,
                                const unsigned chromosomeLength,
                                const unsigned populationCount,
                                const unsigned populationSize,
                                const unsigned eliteSize,
                                const unsigned mutantsSize,
                                const float rhoe) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t newPopSize = populationSize - eliteSize - mutantsSize;
  size_t total = (size_t)populationCount * newPopSize * chromosomeLength;
  if (tid >= total) return;

  auto geneIndex = tid % chromosomeLength;
  auto chromosomeIndex = tid / chromosomeLength % newPopSize + eliteSize;
  // assert(chromosomeIndex < populationSize - mutantsSize);
  auto populationIndex = tid / chromosomeLength / newPopSize;
  // assert(populationIndex < populationCount);
  auto offset = populationIndex * populationSize;
  auto* newChromosome =
      dNewPopulation + (offset + chromosomeIndex) * chromosomeLength;

  auto pid = tid / chromosomeLength + eliteSize;
  auto parentIndex =
      (unsigned)(newChromosome[geneIndex] <= rhoe
                     ? ceilf(dEliteParent[pid] * eliteSize) - 1
                     : eliteSize
                           + ceilf(dParent[pid] * (populationSize - eliteSize))
                           - 1);

  auto* dPopFitnessIdx = dFitnessIdx + offset;
  auto* parent =
      dPopulation + (offset + dPopFitnessIdx[parentIndex]) * chromosomeLength;
  newChromosome[geneIndex] = parent[geneIndex];
}

// no need for mutants
}  // namespace v3

namespace v4 {
__global__ void evolveSurvive(float* dNewPopulation,
                              const float* dPopulation,
                              const unsigned* dFitnessIdx,
                              const unsigned chromosomeLength,
                              const unsigned populationCount,
                              const unsigned populationSize,
                              const unsigned eliteSize) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)populationCount * eliteSize * chromosomeLength;
  if (tid >= total) return;

  auto geneIndex = tid % chromosomeLength;
  auto chromosomeIndex = tid / chromosomeLength % eliteSize;
  auto populationIndex = tid / chromosomeLength / eliteSize;
  auto offset = populationIndex * populationSize;
  auto* newChromosome =
      dNewPopulation + (offset + chromosomeIndex) * chromosomeLength;

  auto* chromosome =
      dPopulation
      + (offset + dFitnessIdx[offset + chromosomeIndex]) * chromosomeLength;
  newChromosome[geneIndex] = chromosome[geneIndex];
}

__global__ void initializeCurandStates(uint n,
                                       curandState_t* states,
                                       const uint seed) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  curand_init(seed, tid, 0, &states[tid]);
}

__device__ void rangeSample(unsigned* sample,
                            unsigned k,
                            unsigned a,
                            unsigned b,
                            curandState_t* state) {
  b -= a;
  for (unsigned i = 0; i < k; ++i) {
    const auto r = curand_uniform(state);
    auto x = (unsigned)ceilf(r * (b - i)) - 1 + a;
    unsigned j;
    for (j = 0; j < i && x >= sample[j]; ++j) ++x;
    for (j = i; j != 0 && x < sample[j - 1]; --j) sample[j] = sample[j - 1];
    sample[j] = x;
  }
}

__global__ void selectParents(unsigned* dParent,
                              const unsigned n,
                              const unsigned numberOfParents,
                              const unsigned numberOfEliteParents,
                              const unsigned populationSize,
                              const unsigned numberOfElites,
                              curandState_t* state) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  const auto nonEliteParents = numberOfParents - numberOfEliteParents;
  rangeSample(dParent + tid * numberOfParents, numberOfEliteParents, 0,
              numberOfElites, &state[tid]);
  rangeSample(dParent + tid * numberOfParents + numberOfEliteParents,
              nonEliteParents, numberOfElites, populationSize, &state[tid]);
}

__global__ void evolveCrossover(float* dNewPopulation,  // random rhoe
                                const float* dPopulation,
                                const unsigned* dFitnessIdx,
                                const unsigned* dParent,
                                const unsigned numberOfParents,
                                const float* dBias,
                                const unsigned chromosomeLength,
                                const unsigned populationCount,
                                const unsigned populationSize,
                                const unsigned eliteSize,
                                const unsigned mutantsSize) {
  extern __shared__ char sharedMemory[];

  auto* bias = (float*)sharedMemory;
  for (unsigned i = threadIdx.x; i < numberOfParents; i += blockDim.x)
    bias[i] = dBias[i];

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t newPopSize = populationSize - eliteSize - mutantsSize;
  size_t total = (size_t)populationCount * newPopSize * chromosomeLength;
  if (tid >= total) return;

  __syncthreads();  // sync to ensure bias was initialized

  auto geneIndex = tid % chromosomeLength;
  auto chromosomeIndex = tid / chromosomeLength % newPopSize + eliteSize;
  // assert(chromosomeIndex < populationSize);
  auto populationIndex = tid / chromosomeLength / newPopSize;
  // assert(populationIndex < populationCount);
  auto offset = populationIndex * populationSize;
  auto* newChromosome =
      dNewPopulation + (offset + chromosomeIndex) * chromosomeLength;

  const auto toss = newChromosome[geneIndex] * bias[numberOfParents - 1];
  unsigned parentIdx = 0;
  while (toss > bias[parentIdx]) {
    // assert(parentIdx < numberOfParents);
    // assert(parentIdx == 0 || bias[parentIdx - 1] < bias[parentIdx]);
    ++parentIdx;
  }

  parentIdx = dParent[(offset + chromosomeIndex) * numberOfParents + parentIdx];
  // assert(parentIdx < populationSize);
  auto* dPopFitnessIdx = dFitnessIdx + offset;
  auto* parent =
      dPopulation + (offset + dPopFitnessIdx[parentIdx]) * chromosomeLength;
  newChromosome[geneIndex] = parent[geneIndex];
}

// no need for mutants
}  // namespace v4

int main() {
  // define params
  const unsigned nLen = 7;  // up to 2^14 = ~8k
  size_t chromosomeLengthsToTest[nLen];
  for (unsigned i = 0; i < nLen; ++i) chromosomeLengthsToTest[i] = 1 << (i + 7);

  const unsigned tests = 10;
  const unsigned maxGenerations = 10000;
  const float rhoe = .7;
  const unsigned numPopulations = 4;
  const unsigned popSize = 256;
  const unsigned numElites = 25;
  const unsigned numMutants = 25;

  cout << "version\tn\tseed\telapsed\tfitness" << endl;

  for (auto chromosomeLength : chromosomeLengthsToTest) {
    const auto totChr = numPopulations * popSize;
    const auto totGenes = totChr * chromosomeLength;

    auto* dPopulation = gpu::alloc<float>(nullptr, totGenes);
    auto* dPopulationTemp = gpu::alloc<float>(nullptr, totGenes);
    auto* dEliteParent = gpu::alloc<float>(nullptr, totGenes);
    auto* dNonEliteParent = gpu::alloc<float>(nullptr, totGenes);
    auto* dFitness = gpu::alloc<float>(nullptr, totChr);
    auto* dFitnessIdx = gpu::alloc<unsigned>(nullptr, totChr);

    gpu::Timer timer;
    for (unsigned t = 1; t <= tests; ++t) {
      // cerr << "Test " << t << endl;
      auto* gen = gpu::allocRandomGenerator(t);

      gpu::random(nullptr, gen, dPopulation, totGenes);
      decode(dFitness, dFitnessIdx, dPopulation, numPopulations, popSize,
             chromosomeLength);
      gpu::sync();

      timer.reset();
      for (unsigned g = 1; g <= maxGenerations; ++g) {
        gpu::random(nullptr, gen, dPopulationTemp, totGenes);
        gpu::random(nullptr, gen, dEliteParent, totGenes);
        gpu::random(nullptr, gen, dNonEliteParent, totGenes);

        const auto threads = THREADS_PER_BLOCK;
        const auto blocks = (numPopulations * popSize + threads - 1) / threads;
        v0::evolve<<<blocks, threads>>>(
            dPopulationTemp, dPopulation, dFitnessIdx, dEliteParent,
            dNonEliteParent, chromosomeLength, numPopulations, popSize,
            numElites, numMutants, rhoe);

        swap(dPopulation, dPopulationTemp);
        decode(dFitness, dFitnessIdx, dPopulation, numPopulations, popSize,
               chromosomeLength);
      }

      gpu::sync();
      const auto elapsed = timer.seconds();
      const auto fitness =
          getBest(dFitness, dFitnessIdx, numPopulations, popSize);
      cout << fixed << setprecision(6) << "v0"
           << "\t" << chromosomeLength << "\t" << t << "\t" << elapsed << "\t"
           << fitness << endl;
      gpu::free(gen);
    }

    gpu::free(nullptr, dPopulation);
    gpu::free(nullptr, dPopulationTemp);
    gpu::free(nullptr, dEliteParent);
    gpu::free(nullptr, dNonEliteParent);
    gpu::free(nullptr, dFitness);
    gpu::free(nullptr, dFitnessIdx);
  }

  for (auto chromosomeLength : chromosomeLengthsToTest) {
    const auto totChr = numPopulations * popSize;
    const auto totGenes = totChr * chromosomeLength;

    auto* dPopulation = gpu::alloc<float>(nullptr, totGenes);
    auto* dPopulationTemp = gpu::alloc<float>(nullptr, totGenes);
    auto* dEliteParent = gpu::alloc<float>(nullptr, totGenes);
    auto* dNonEliteParent = gpu::alloc<float>(nullptr, totGenes);
    auto* dFitness = gpu::alloc<float>(nullptr, totChr);
    auto* dFitnessIdx = gpu::alloc<unsigned>(nullptr, totChr);

    gpu::Timer timer;
    for (unsigned t = 1; t <= tests; ++t) {
      // cerr << "Test " << t << endl;
      auto* gen = gpu::allocRandomGenerator(t);

      gpu::random(nullptr, gen, dPopulation, totGenes);
      decode(dFitness, dFitnessIdx, dPopulation, numPopulations, popSize,
             chromosomeLength);
      gpu::sync();

      timer.reset();
      for (unsigned g = 1; g <= maxGenerations; ++g) {
        gpu::random(nullptr, gen, dPopulationTemp, totGenes);
        gpu::random(nullptr, gen, dEliteParent, totGenes);
        gpu::random(nullptr, gen, dNonEliteParent, totGenes);

        const auto threads = THREADS_PER_BLOCK;
        const auto blocks =
            (numPopulations * popSize * chromosomeLength + threads - 1)
            / threads;
        v1::evolve<<<blocks, threads>>>(
            dPopulationTemp, dPopulation, dFitnessIdx, dEliteParent,
            dNonEliteParent, chromosomeLength, numPopulations, popSize,
            numElites, numMutants, rhoe);

        swap(dPopulation, dPopulationTemp);
        decode(dFitness, dFitnessIdx, dPopulation, numPopulations, popSize,
               chromosomeLength);
      }

      gpu::sync();
      const auto elapsed = timer.seconds();
      const auto fitness =
          getBest(dFitness, dFitnessIdx, numPopulations, popSize);
      cout << fixed << setprecision(6) << "v1"
           << "\t" << chromosomeLength << "\t" << t << "\t" << elapsed << "\t"
           << fitness << endl;
      gpu::free(gen);
    }

    gpu::free(nullptr, dPopulation);
    gpu::free(nullptr, dPopulationTemp);
    gpu::free(nullptr, dEliteParent);
    gpu::free(nullptr, dNonEliteParent);
    gpu::free(nullptr, dFitness);
    gpu::free(nullptr, dFitnessIdx);
  }

  for (auto chromosomeLength : chromosomeLengthsToTest) {
    const auto totChr = numPopulations * popSize;
    const auto totGenes = totChr * chromosomeLength;

    auto* dPopulation = gpu::alloc<float>(nullptr, totGenes);
    auto* dPopulationTemp = gpu::alloc<float>(nullptr, totGenes);
    auto* dEliteParent = gpu::alloc<float>(nullptr, totGenes);
    auto* dNonEliteParent = gpu::alloc<float>(nullptr, totGenes);
    auto* dFitness = gpu::alloc<float>(nullptr, totChr);
    auto* dFitnessIdx = gpu::alloc<unsigned>(nullptr, totChr);

    gpu::Timer timer;
    for (unsigned t = 1; t <= tests; ++t) {
      // cerr << "Test " << t << endl;
      auto* gen = gpu::allocRandomGenerator(t);

      gpu::random(nullptr, gen, dPopulation, totGenes);
      decode(dFitness, dFitnessIdx, dPopulation, numPopulations, popSize,
             chromosomeLength);
      gpu::sync();

      timer.reset();
      for (unsigned g = 1; g <= maxGenerations; ++g) {
        gpu::random(nullptr, gen, dPopulationTemp, totGenes);
        gpu::random(nullptr, gen, dEliteParent, totGenes);
        gpu::random(nullptr, gen, dNonEliteParent, totGenes);

        const auto threads = THREADS_PER_BLOCK;
        const auto blocks =
            (numPopulations * popSize * chromosomeLength + threads - 1)
            / threads;
        v2::evolve<<<blocks, threads>>>(
            dPopulationTemp, dPopulation, dFitnessIdx, dEliteParent,
            dNonEliteParent, chromosomeLength, numPopulations, popSize,
            numElites, numMutants, rhoe);

        swap(dPopulation, dPopulationTemp);
        decode(dFitness, dFitnessIdx, dPopulation, numPopulations, popSize,
               chromosomeLength);
      }

      gpu::sync();
      const auto elapsed = timer.seconds();
      const auto fitness =
          getBest(dFitness, dFitnessIdx, numPopulations, popSize);
      cout << fixed << setprecision(6) << "v2"
           << "\t" << chromosomeLength << "\t" << t << "\t" << elapsed << "\t"
           << fitness << endl;
      gpu::free(gen);
    }

    gpu::free(nullptr, dPopulation);
    gpu::free(nullptr, dPopulationTemp);
    gpu::free(nullptr, dEliteParent);
    gpu::free(nullptr, dNonEliteParent);
    gpu::free(nullptr, dFitness);
    gpu::free(nullptr, dFitnessIdx);
  }

  for (auto chromosomeLength : chromosomeLengthsToTest) {
    const auto totChr = numPopulations * popSize;
    const auto totGenes = totChr * chromosomeLength;

    auto* dPopulation = gpu::alloc<float>(nullptr, totGenes);
    auto* dPopulationTemp = gpu::alloc<float>(nullptr, totGenes);
    auto* dEliteParent = gpu::alloc<float>(nullptr, totGenes);
    auto* dNonEliteParent = gpu::alloc<float>(nullptr, totGenes);
    auto* dFitness = gpu::alloc<float>(nullptr, totChr);
    auto* dFitnessIdx = gpu::alloc<unsigned>(nullptr, totChr);

    gpu::Timer timer;
    for (unsigned t = 1; t <= tests; ++t) {
      // cerr << "Test " << t << endl;
      auto* gen = gpu::allocRandomGenerator(t);

      gpu::random(nullptr, gen, dPopulation, totGenes);
      decode(dFitness, dFitnessIdx, dPopulation, numPopulations, popSize,
             chromosomeLength);
      gpu::sync();

      timer.reset();
      for (unsigned g = 1; g <= maxGenerations; ++g) {
        gpu::random(nullptr, gen, dPopulationTemp, totGenes);
        gpu::random(nullptr, gen, dEliteParent, totGenes);
        gpu::random(nullptr, gen, dNonEliteParent, totGenes);

        const auto threads = THREADS_PER_BLOCK;
        auto blocks =
            (numPopulations * numElites * chromosomeLength + threads - 1)
            / threads;
        v3::evolveSurvive<<<blocks, threads>>>(
            dPopulationTemp, dPopulation, dFitnessIdx, chromosomeLength,
            numPopulations, popSize, numElites);

        blocks = (numPopulations * (popSize - numElites - numMutants)
                      * chromosomeLength
                  + threads - 1)
                 / threads;
        v3::evolveCrossover<<<blocks, threads>>>(
            dPopulationTemp, dPopulation, dFitnessIdx, dEliteParent,
            dNonEliteParent, chromosomeLength, numPopulations, popSize,
            numElites, numMutants, rhoe);

        swap(dPopulation, dPopulationTemp);
        decode(dFitness, dFitnessIdx, dPopulation, numPopulations, popSize,
               chromosomeLength);
      }

      gpu::sync();
      const auto elapsed = timer.seconds();
      const auto fitness =
          getBest(dFitness, dFitnessIdx, numPopulations, popSize);
      cout << fixed << setprecision(6) << "v3"
           << "\t" << chromosomeLength << "\t" << t << "\t" << elapsed << "\t"
           << fitness << endl;
      gpu::free(gen);
    }

    gpu::free(nullptr, dPopulation);
    gpu::free(nullptr, dPopulationTemp);
    gpu::free(nullptr, dEliteParent);
    gpu::free(nullptr, dNonEliteParent);
    gpu::free(nullptr, dFitness);
    gpu::free(nullptr, dFitnessIdx);
  }

  std::vector<std::vector<float>> biases;

  biases.push_back({});
  biases.back().push_back(rhoe);
  biases.back().push_back(1 - rhoe);

  // biases.push_back({});
  // for (unsigned i = 1; i <= 5; ++i) biases.back().push_back(1.0f / (float)i);

  for (unsigned i = 0; i < biases.size(); ++i)
    for (unsigned j = 1; j < biases[i].size(); ++j)
      biases[i][j] += biases[i][j - 1];

  for (const auto& bias : biases) {
    for (auto chromosomeLength : chromosomeLengthsToTest) {
      const unsigned numParents = bias.size();
      const unsigned numEliteParents = numParents / 2;
      const auto totChr = numPopulations * popSize;
      const auto totGenes = totChr * chromosomeLength;

      auto* dPopulation = gpu::alloc<float>(nullptr, totGenes);
      auto* dPopulationTemp = gpu::alloc<float>(nullptr, totGenes);
      auto* dBias = gpu::alloc<float>(nullptr, numParents);
      auto* dParent = gpu::alloc<unsigned>(nullptr, totChr * numParents);
      auto* dRandomStates =
          gpu::alloc<curandState_t>(nullptr, totChr * numParents);
      auto* dFitness = gpu::alloc<float>(nullptr, totChr);
      auto* dFitnessIdx = gpu::alloc<unsigned>(nullptr, totChr);

      gpu::copy2d(nullptr, dBias, bias.data(), bias.size());

      vector<float> elapsed;
      vector<float> fitness;
      gpu::Timer timer;
      for (unsigned t = 1; t <= tests; ++t) {
        // cerr << "Test " << t << endl;
        auto* gen = gpu::allocRandomGenerator(t);

        v4::initializeCurandStates<<<
            (totChr * numParents + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
            THREADS_PER_BLOCK>>>(totChr * numParents, dRandomStates,
                                 /* seed: */ t);

        gpu::random(nullptr, gen, dPopulation, totGenes);
        decode(dFitness, dFitnessIdx, dPopulation, numPopulations, popSize,
               chromosomeLength);
        gpu::sync();

        timer.reset();
        for (unsigned g = 1; g <= maxGenerations; ++g) {
          gpu::random(nullptr, gen, dPopulationTemp, totGenes);

          const auto threads = THREADS_PER_BLOCK;
          auto blocks = (numPopulations * popSize + threads - 1) / threads;
          v4::selectParents<<<blocks, threads>>>(
              dParent, numPopulations * popSize, numParents, numEliteParents,
              popSize, numElites, dRandomStates);

          blocks = (numPopulations * numElites * chromosomeLength + threads - 1)
                   / threads;
          v4::evolveSurvive<<<blocks, threads>>>(
              dPopulationTemp, dPopulation, dFitnessIdx, chromosomeLength,
              numPopulations, popSize, numElites);

          blocks = (numPopulations * (popSize - numElites - numMutants)
                        * chromosomeLength
                    + threads - 1)
                   / threads;
          const auto sharedMemSize = numParents * sizeof(float);
          v4::evolveCrossover<<<blocks, threads, sharedMemSize>>>(
              dPopulationTemp, dPopulation, dFitnessIdx, dParent, numParents,
              dBias, chromosomeLength, numPopulations, popSize, numElites,
              numMutants);

          swap(dPopulation, dPopulationTemp);
          decode(dFitness, dFitnessIdx, dPopulation, numPopulations, popSize,
                 chromosomeLength);
        }

        gpu::sync();
        const auto elapsed = timer.seconds();
        const auto fitness =
            getBest(dFitness, dFitnessIdx, numPopulations, popSize);
        cout << fixed << setprecision(6);
        // show bias instead of the version
        cout << "bias:";
        for (const auto x : bias) cout << " " << x;
        cout << "\t" << chromosomeLength << "\t" << t << "\t" << elapsed << "\t"
             << fitness << endl;
        gpu::free(gen);
      }

      gpu::free(nullptr, dPopulation);
      gpu::free(nullptr, dPopulationTemp);
      gpu::free(nullptr, dBias);
      gpu::free(nullptr, dParent);
      gpu::free(nullptr, dRandomStates);
      gpu::free(nullptr, dFitness);
      gpu::free(nullptr, dFitnessIdx);
    }
  }

  return 0;
}
