#include "Random.cl"
#include "Algorithm.cl"
#include "Fitness.cl"

kernel void buildPopulation(
    int popSize,
    int chromosomeLength,
    global float* population,
    global uint* seeds) {
  const int i = get_global_id(0);
  if (i >= popSize) return;
  for (int k = 0; k < chromosomeLength; ++k) population[k * popSize + i] = random(seeds);
}

/**
 * Return the genes of the chromosomes representing the order of the gene value.
 * @param popSize The number of chromosomes in the population.
 * @param chromosomeLength The number of genes in each chromosome.
 * @param genes Used to save the genes of the genes.
 * @param population Current generation of chromosomes.
 * @param indices The indices to sort.
 * @param indicesTmp Temporary memory for the indices.
 */
kernel void buildPopulationIndices(
    int popSize,
    int chromosomeLength,
    global const float* population,
    global int* genes,
    global int* genesTmp) {
  int i = get_global_id(0);
  if (i >= popSize) return;
  for (int k = 0; k < chromosomeLength; ++k) genes[k * popSize + i] = k;
  sortColumnsByKey(popSize, chromosomeLength, population, genes, genesTmp);
}

/**
 * Set the fitness of each chromosome, sorting them in ascending order.
 * @param popSize The number of chromosomes in the population.
 * @param fitness Used to return the new fitness.
 * @param newFitness The values of the fitness to set.
 */
kernel void setPopulation(
    int popSize,
    int chromosomeLength,
    global float* population,
    global const float* newPopulation,
    global const int* order) {
  const int i = get_global_id(0);
  if (i >= popSize) return;

  const int end = popSize * chromosomeLength;
  const int steps = popSize;
  const int ii = order[i];
  for (int j = 0; j < end; j += steps)
    population[j + i] = newPopulation[j + ii];
}

/**
 * Performs a single evolution in the population (moves to a new generation).
 * @param popSize The number of chromosomes in the population.
 * @param chromosomeLength The number of genes in each chromosome.
 * @param fitness The fitness of the current generation.
 * @param population Current generation of chromosomes.
 * @param newPopulation Used to return the population.
 * @param eliteSize The number of members to consider elite.
 * @param mutantSize The number of members to discard and regenerate randomly.
 * @param rho The probability to select a gene from an elite member.
 * @param seeds Used to select the genes for the population.
 * @warning Should be used with @p eliteSize > 0, @p mutantSize > 0, and @p rho > 0.5.
 * @warning This function assumes that the chromosomes in @p population are in increasing order of fitness.
 */
kernel void evolvePopulation(
    int popSize,
    int chromosomeLength,
    global float* population,
    global float* newPopulation,
    int eliteSize,
    int mutantSize,
    float rho,
    global uint* seeds) {
  const int i = get_global_id(0);
  const int end = popSize * chromosomeLength;
  const int steps = popSize;
  if (i < eliteSize) {
    // copy elite
    for (int j = i; j < end; j += steps)
      newPopulation[j] = population[j];
  } else if (i < popSize - mutantSize) {
    // crossover
    const int eliteParent = randrange(0, eliteSize, seeds);
    const int nonEliteParent = randrange(eliteSize, popSize, seeds);
    for (int j = 0; j < end; j += steps)
      newPopulation[j + i] = random(seeds) <= rho ? population[j + eliteParent] : population[j + nonEliteParent];
  } else if (i < popSize) {
    // mutants
    for (int j = i; j < end; j += steps)
      newPopulation[j] = random(seeds);
  }
}

kernel void replaceWorst(
    int popSize,
    int chromosomeLength,
    global float* population,
    int totalReplaced,
    global float* newChromosomes) {
  const int tid = get_global_id(0);
  if (tid >= totalReplaced) return;

  const int src = tid;
  const int dst = (tid < totalReplaced
      ? tid  // replace the elite members to avoid duplicating the chromosomes
      : popSize - totalReplaced + tid);

  for (int j = 0; j < chromosomeLength; ++j)
    population[j * popSize + dst] = newChromosomes[j * totalReplaced + src];
}
