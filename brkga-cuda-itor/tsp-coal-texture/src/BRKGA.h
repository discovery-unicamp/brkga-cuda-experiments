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

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "Decoder.h"
#include "CommonStructs.h"
#include "ConfigFile.h"
#include <omp.h>

#define THREADS_PER_BLOCK 256

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)
// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaDeviceSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)


class BRKGA{
public:
	BRKGA(unsigned n, unsigned p, float pe, float pm, float rhoe, unsigned K, unsigned decode_type, unsigned NUM_THREADS=1, unsigned RAND_SEED=1234);
	~BRKGA();
	void reset_population(void);
	void evolve(int number_generations=1);
	void exchangeElite(unsigned M);
	std::vector<std::vector <float>> getkBestChromosomes(unsigned k);
	void setInstanceInfo(void *info, long unsigned num, long unsigned size);
	void setInstanceInfo2D(float *info, long unsigned columns, long unsigned rows, long unsigned size);
  void saveBestChromosomes();
  std::vector<std::vector <float>> getkBestChromosomes2(unsigned k);


private:
	float *d_population=NULL; //device population
	float *d_population2=NULL; //device population
	float *h_population=NULL; //host population

	float *d_scores=NULL; //device score of each chromossome
	float *h_scores=NULL; //host score of each chromossome

	void *d_instance_info=NULL; //vector of unknow type containg instance info used to evaluate a chromosome
	void *h_instance_info=NULL;

	float *d_best_solutions=NULL; //pool of 10 best solutions
	float *h_best_solutions=NULL;
	unsigned best_saved=0; //indicate whether best solutions were already saved or not

	PopIdxThreadIdxPair *d_scores_idx=NULL; //table with indices of each chromosome score on device 
	PopIdxThreadIdxPair *h_scores_idx=NULL; //table with indices of each chromosome score on host

	ChromosomeGeneIdxPair *d_chromosome_gene_idx = NULL; //table with indices for all chromosomes and each of its gene on device
	ChromosomeGeneIdxPair *h_chromosome_gene_idx = NULL; //table with indices for all chromosomes and each of its gene on host


	float *d_random_elite_parent=NULL; //a random number for each thread to choose its elite parent during crossover
	float *d_random_parent=NULL; //a random number for each thread to choose its non-elite parent during crossover

	unsigned number_chromosomes;//total number of chromosomes in all K populations
	unsigned number_genes;//total number of genes in all K populations
	unsigned chromosome_size;
	unsigned population_size;
	unsigned elite_size;
	unsigned mutants_size;
	unsigned number_populations;
	float rhoe;

	curandGenerator_t gen; //cuda ramdom number generator
	dim3 dimBlock;
  dim3 dimGrid;
  dim3 dimGridChromo;
  dim3 dimGrid_population;

  unsigned decode_type;


  unsigned OMP_THREADS=8; //if host_decod is used openmp can be used to decode

	size_t pitch;

  void initialize_population(int p);
	void global_sort_chromosomes();
	void sort_chromosomes();
	void sort_chromosomes_genes();
	void evaluate_chromosomes_host();
  void evaluate_chromosomes_device();
  void evaluate_chromosomes_sorted_device();
  void evaluate_chromosomes_sorted_device_texture();
  void evaluate_chromosomes_sorted_device_coalesced();
  void test_memory_malloc(cudaError_t err, unsigned code, unsigned total_memory);
};

#endif
