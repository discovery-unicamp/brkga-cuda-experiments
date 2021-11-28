#include "Brkga.hpp"

Brkga::Brkga(const cl::Device& device, Problem* _problem, const Configuration& config) :
    numberOfPopulations(config.geti("number-of-populations")),
    populationSize(config.geti("population-size")),
    chromosomeLength(_problem->chromosomeLength()),
    eliteLength(config.geti("elite-size")),
    mutantsLength(config.geti("mutants-size")),
    rho(config.getf("rho")),
    currentGeneration(0),
    exchangeBestInterval(config.geti("exchange-best-interval")),
    exchangeBestSize(config.geti("exchange-best-size")),
    bestFitness((float) 1e50),  // NOLINT
    problem(_problem),
    kernel(device, config.get("opencl-flags").data()),
    threadsPerBlock(config.geti("threads-per-block")),
    dSeeds(numberOfPopulations),
    dPopulations(numberOfPopulations),
    dPopulationsTemp(numberOfPopulations),
    hPopulationsIndicesTemp(numberOfPopulations),
    dIndicesTemp(numberOfPopulations),
    dIndicesSortTemp(numberOfPopulations) {
  assert(numberOfPopulations > 0);
  assert(eliteLength > 0);
  assert(mutantsLength >= 0);
  assert(eliteLength + mutantsLength < populationSize);
  assert(chromosomeLength > 0);
  assert(rho > minimumRhoValue);  // give priority to elite parent
  assert(rho < 1.0);  // avoid simply copying the elite member
  assert(exchangeBestInterval == 0 || numberOfPopulations > 1);  // must have multiple populations to exchange
  assert(exchangeBestInterval == 0 || exchangeBestSize > 0);  // must exchange something

  if (mutantsLength == 0) warn("Population will evolve without mutation");

  info("Allocating memory");
  std::mt19937 rng(config.geti("seed"));
  std::uniform_int_distribution uid;
  std::vector<unsigned> hSeeds(populationSize);
  for (int i = 0; i < numberOfPopulations; ++i) {
    dPopulations[i] = kernel.buffer<float>(populationSize * chromosomeLength);
    dPopulationsTemp[i] = kernel.buffer<float>(populationSize * chromosomeLength);
    hPopulationsIndicesTemp[i].resize(populationSize * chromosomeLength);
    dIndicesTemp[i] = kernel.buffer<int>(populationSize * chromosomeLength);
    dIndicesSortTemp[i] = kernel.buffer<int>(populationSize * chromosomeLength);

    std::generate(hSeeds.begin(), hSeeds.end(), [&uid, &rng]() { return uid(rng); });
    dSeeds[i] = kernel.buffer(hSeeds);
  }

  info("Generating population");
  kernel.startPipeline()
      .then(numberOfPopulations, [&](auto p, auto& pipe) {
        pipe.then(kernel.buildPopulation(populationSize, chromosomeLength, dPopulationsTemp[p], dSeeds[p]))
            .run(populationSize, threadsPerBlock);
      }).wait();

  assignPopulationFromTemp();
  validate();

  info("Best initial chromosome:", bestFitness);
}

void Brkga::evolve() {
  if (exchangeBestInterval != 0
      && currentGeneration % exchangeBestInterval == 0
      && numberOfPopulations != 1
      && currentGeneration != 0)
    exchangeBestChromosomes();

  debug("Evolving generation", currentGeneration);
  kernel.startPipeline()
      .then(numberOfPopulations, [&](auto p, auto& pipe) {
        pipe.then(kernel.evolvePopulation(populationSize, chromosomeLength, dPopulations[p], dPopulationsTemp[p],
                                          eliteLength, mutantsLength, rho, dSeeds[p]))
            .run(populationSize, threadsPerBlock);
      }).wait();

  assignPopulationFromTemp();
  ++currentGeneration;
  validate();

  info(
#if LOG_LEVEL == 1
      LoggerOptions().end("\r"),
#endif //LOG_LEVEL
    "Generation", currentGeneration, "has fitness", getBestFitness());
}

void Brkga::exchangeBestChromosomes() {
  debug("Exchanging the best", exchangeBestSize, "chromosomes");

  const int totalExchanged = numberOfPopulations * exchangeBestSize;
  assert(exchangeBestSize <= eliteLength);  // exchange only elites
  assert(totalExchanged <= populationSize);  // exchanged members cannot exceed population size

  // TODO will this be faster in the kernel?
  std::vector<float> bestChromosomes(totalExchanged * chromosomeLength);
  for (int p = 0; p < numberOfPopulations; ++p) {
    auto hPopulation = kernel.read<float>(dPopulations[p], 0, populationSize * chromosomeLength);
    for (int i = 0; i < exchangeBestSize; ++i)
      for (int j = 0; j < chromosomeLength; ++j)
        bestChromosomes[j * totalExchanged + i + p * exchangeBestSize] = hPopulation[j * populationSize + i];
  }

  auto dBestChromosomes = kernel.buffer(bestChromosomes, true);
  kernel.startPipeline()
      .then(numberOfPopulations, [&](auto p, auto& pipe) {
        pipe.then(kernel.replaceWorst(populationSize, chromosomeLength, dPopulations[p], totalExchanged,
                                      dBestChromosomes))
            .run(totalExchanged, threadsPerBlock);
      }).wait();

  std::swap(dPopulations, dPopulationsTemp);  // the next method will copy from temp
  assignPopulationFromTemp();
  validate();
}

void Brkga::assignPopulationFromTemp() {
  // for host decode, it reads to host memory
  kernel.startPipeline()
      .then(numberOfPopulations, [&](auto p, auto& pipe) {
        pipe.then(kernel.buildPopulationIndices(populationSize, chromosomeLength, dPopulationsTemp[p], dIndicesTemp[p],
                                                dIndicesSortTemp[p]))
            .run(populationSize, threadsPerBlock)
            .thenReadFromTo(dIndicesTemp[p], hPopulationsIndicesTemp[p].data(), 0, populationSize * chromosomeLength);
      }).wait();

  kernel.startPipeline()
      .then(numberOfPopulations, [&](auto p, auto& pipe) {
        std::vector<int> indices(chromosomeLength);
        std::vector<float> fitness(populationSize);
        for (int i = 0; i < populationSize; ++i) {
          for (int j = 0; j < chromosomeLength; ++j)
            indices[j] = hPopulationsIndicesTemp[p][j * populationSize + i];
          assert((int) std::set(indices.begin(), indices.end()).size() == chromosomeLength);  // missing or duplicated
          fitness[i] = problem->evaluateIndices(indices.data());
        }
        trace("New fitness:", fitness);

        std::vector<int> order(populationSize);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&fitness](int lhs, int rhs) { return fitness[lhs] < fitness[rhs]; });
        trace("Fitness order", order);
        assert((int) std::set(order.begin(), order.end()).size() == populationSize);  // missing or duplicated fitness
        bestFitness = std::min(bestFitness, fitness[order[0]]);

        pipe.then(kernel.setPopulation(populationSize, chromosomeLength, dPopulations[p], dPopulationsTemp[p],
                                       kernel.buffer(order, true)))
            .run(populationSize, threadsPerBlock);
      }).wait();
}

void Brkga::validate() {
#ifndef NDEBUG
  debug("validating");
  assert(numberOfPopulations > 0);  // should have at least one member
  assert(populationSize > 0);  // should have chromosomes
  assert(chromosomeLength > 0);
  assert(eliteLength > 0);  // should have elite members
  assert(eliteLength < populationSize);  // population cannot be elite
  assert(mutantsLength >= 0);
  assert(mutantsLength < populationSize);  // population cannot be mutants
  assert(eliteLength + mutantsLength < populationSize);  // will not generate new chromosomes
  assert(rho > minimumRhoValue);  // should give preference to elite
  assert(rho < 1.0);  // will simply copy the elite
  assert(currentGeneration >= 0);

  float expectedBestFitness = -1;
  for (int p = 0; p < numberOfPopulations; ++p) {
    kernel.startPipeline()
        .then(kernel.buildPopulationIndices(populationSize, chromosomeLength, dPopulations[p], dIndicesTemp[p],
                                            dIndicesSortTemp[p]))
        .run(populationSize, threadsPerBlock)
        .wait();
    auto population = kernel.read<float>(dPopulations[p], 0, populationSize * chromosomeLength);
    auto populationIndices = kernel.read<int>(dIndicesTemp[p], 0, populationSize * chromosomeLength);

    std::vector<float> fitness(populationSize);
    for (int i = 0; i < populationSize; ++i) {
      std::vector<float> chromosome(chromosomeLength);
      for (int j = 0; j < chromosomeLength; ++j)
        chromosome[j] = population[j * populationSize + i];
      trace("Chromosome", i, chromosome);
      assert((int) std::set(chromosome.begin(), chromosome.end()).size() >= chromosomeLength / 2);  // many copies
      assert(std::all_of(chromosome.begin(), chromosome.end(), [](float a) { return a >= 0; }));
      assert(std::all_of(chromosome.begin(), chromosome.end(), [](float a) { return a < 1; }));

      std::vector<int> indices(chromosomeLength);
      for (int j = 0; j < chromosomeLength; ++j)
        indices[j] = populationIndices[j * populationSize + i];
      trace("Indices", i, indices);
      assert(*std::min_element(indices.begin(), indices.end()) >= 0);
      assert(*std::max_element(indices.begin(), indices.end()) < chromosomeLength);
      assert((int) std::set(indices.begin(), indices.end()).size() == chromosomeLength);  // duplicated genes

      std::vector<float> sortedChromosome(chromosomeLength);
      for (int k = 0; k < chromosomeLength; ++k)
        sortedChromosome[k] = chromosome[indices[k]];
      trace("Sorted chromosome", i, sortedChromosome);
      for (int k = 1; k < chromosomeLength; ++k)
        assert(sortedChromosome[k - 1] <= sortedChromosome[k]);

      fitness[i] = problem->evaluateIndices(indices.data());
    }

    assert(fitness[0] > 0);
    for (int i = 1; i < populationSize; ++i)
      assert(fitness[i - 1] <= fitness[i]);

    if (p == 0 || fitness[0] < expectedBestFitness)
      expectedBestFitness = fitness[0];
  }

  // NOLINTNEXTLINE
  assert(std::abs(bestFitness - expectedBestFitness) < eps);  // should save the best fitness
#endif  // NDEBUG
}
