/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */

#ifndef BRKGA_H
#define BRKGA_H

#include <stdio.h>
#include <iostream>
#include <exception>
#include <vector>

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "Decoder.h"
#include "CommonStructs.h"
#include "ConfigFile.h"
#include <omp.h>

#define THREADS_PER_BLOCK 256


class BRKGA{
public:
	BRKGA(unsigned n, unsigned p, float pe, float pm, float rhoe, unsigned K, unsigned decode_type, unsigned NUM_THREADS=1, unsigned RAND_SEED=1234);
	~BRKGA();
	void reset_population(void);
	void evolve(int number_generations=1);
	void exchangeElite(unsigned M);
	std::vector<std::vector <float>> getkBestChromosomes(unsigned k);
	void setInstanceInfo(void *info, long unsigned num, long unsigned size);
  void saveBestChromosomes();
  std::vector<std::vector <float>> getkBestChromosomes2(unsigned k);


private:
	float *d_population=NULL; /// device population pointer
	float *d_population2=NULL; /// device population pointer
	float *h_population=NULL; /// host population pointer

	float *d_scores=NULL; /// device score pointer to save score of each chromossome
	float *h_scores=NULL; /// host score pointer to save score of each chromossome

	void *d_instance_info=NULL; /// array of unknow type in the device containg instance info used to evaluate a chromosome
	void *h_instance_info=NULL; /// array of unknow type in the host containg instance info used to evaluate a chromosome

	float *d_best_solutions=NULL; /// device pool of POOL_SIZE (see ConfigFile.h) best solutions
	float *h_best_solutions=NULL; /// device pool of POOL_SIZE (see ConfigFile.h) best solutions
	unsigned best_saved=0; //indicate whether best solutions were already saved or not

	PopIdxThreadIdxPair *d_scores_idx=NULL; /// table with indices of each chromosome score on device 
	PopIdxThreadIdxPair *h_scores_idx=NULL; /// table with indices of each chromosome score on host

	ChromosomeGeneIdxPair *d_chromosome_gene_idx = NULL; /// table with indices for all chromosomes and each of its genes on device
	ChromosomeGeneIdxPair *h_chromosome_gene_idx = NULL; /// table with indices for all chromosomes and each of its genes on host


	float *d_random_elite_parent=NULL; /// contains a random number for each thread to choose its elite parent during crossover
	float *d_random_parent=NULL; /// contains a random number for each thread to choose its non-elite parent during crossover

	unsigned number_chromosomes;/// total number of chromosomes in all K populations
	unsigned chromosome_size;
	unsigned population_size;
	unsigned elite_size;
	unsigned mutants_size;
	unsigned number_populations;
	float rhoe; /// parameter to chose gene from elite parent or non-elite parent

	curandGenerator_t gen; //cuda ramdom number generator
	dim3 dimBlock;
  dim3 dimGrid;

  unsigned decode_type;


  unsigned NUM_THREADS=8; //if host_decod is used openmp can be used to decode


  void initialize_population(int p);
	void global_sort_chromosomes();
	void sort_chromosomes();
	void sort_chromosomes_genes();
	void evaluate_chromosomes_host();
  void evaluate_chromosomes_device();
  void evaluate_chromosomes_sorted_device();
  void test_memory_malloc(cudaError_t err, unsigned code, unsigned total_memory);
};

#endif
