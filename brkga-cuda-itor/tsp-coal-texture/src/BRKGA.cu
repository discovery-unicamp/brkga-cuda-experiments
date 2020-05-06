/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */


#include "BRKGA.h"

texture<float, 2, cudaReadModeElementType> tex;

/***
Constructor
***/
BRKGA::BRKGA(unsigned n, unsigned p, float pe, float pm, float rhoe, unsigned K, unsigned decode_type, unsigned OMP_THREADS, unsigned RAND_SEED){
	this->population_size = p;
	this->number_populations = K;
	this->number_chromosomes = p * K; 
	this->number_genes = this->number_chromosomes*n;
	this->chromosome_size = n;
	this->elite_size = (unsigned)(pe*p);
	this->mutants_size = (unsigned)(pm*p);
	this->rhoe = rhoe;
	this->decode_type = decode_type;
	this->OMP_THREADS = OMP_THREADS;

	using std::range_error;
	if(chromosome_size == 0) { throw range_error("Chromosome size equals zero."); }
	if(population_size == 0) { throw range_error("Population size equals zero."); }
	if(elite_size == 0) { throw range_error("Elite-set size equals zero."); }
	if(elite_size + mutants_size > population_size) { throw range_error("elite + mutant sets greater than population size (p)."); }
	if(number_populations == 0) { throw range_error("Number of parallel populations cannot be zero."); }

	long unsigned total_memory=0;
	// Allocate a float array representing all K populations on host and device
	h_population = (float *)malloc(number_chromosomes*chromosome_size*sizeof(float));
	total_memory += number_chromosomes*chromosome_size*sizeof(float);
	test_memory_malloc(cudaMalloc((void **)&d_population, number_chromosomes*chromosome_size*sizeof(float)), 1, total_memory);

	total_memory += number_chromosomes*chromosome_size*sizeof(float);
	test_memory_malloc(cudaMalloc((void **)&d_population2, number_chromosomes*chromosome_size*sizeof(float)), 2, total_memory);

	total_memory += number_chromosomes*sizeof(float);
	   // Allocate an array representing the scores of each chromosome on host and device
	h_scores = (float *)malloc(number_chromosomes*sizeof(float));
	test_memory_malloc(cudaMalloc((void **)&d_scores, number_chromosomes*sizeof(float)), 3, total_memory);

	total_memory += number_chromosomes*sizeof(PopIdxThreadIdxPair);
	   // Allocate an array representing the indices of each chromosome on host and device
	h_scores_idx = (PopIdxThreadIdxPair *)malloc(number_chromosomes*sizeof(PopIdxThreadIdxPair));
	test_memory_malloc(cudaMalloc((void **)&d_scores_idx, number_chromosomes*sizeof(PopIdxThreadIdxPair)), 4, total_memory);

	total_memory += number_chromosomes*chromosome_size*sizeof(ChromosomeGeneIdxPair);
	   // Allocate an array representing the indices of each gene of each chromosome on host and device
	h_chromosome_gene_idx = (ChromosomeGeneIdxPair *)malloc(number_chromosomes*chromosome_size*sizeof(ChromosomeGeneIdxPair));
	test_memory_malloc(cudaMalloc((void **)&d_chromosome_gene_idx, number_chromosomes*chromosome_size*sizeof(ChromosomeGeneIdxPair)), 5, total_memory);

	total_memory += number_chromosomes*sizeof(float);
	test_memory_malloc(cudaMalloc((void **)&d_random_elite_parent, number_chromosomes*sizeof(float)), 6, total_memory);

	total_memory += number_chromosomes*sizeof(float);
	test_memory_malloc(cudaMalloc((void **)&d_random_parent, number_chromosomes*sizeof(float)), 7, total_memory);

	// Allocate a poll to save the POOL_SIZE best solutions, where the first value in each chromosome is the chromosome score
	h_best_solutions = (float *)malloc(POOL_SIZE*(chromosome_size+1)*sizeof(float));
	test_memory_malloc(cudaMalloc((void **)&d_best_solutions, POOL_SIZE*(chromosome_size+1)*sizeof(float)), 8, total_memory);

	printf("Total Memory Used In GPU %lu bytes(%lu Mbytes)\n", total_memory, total_memory/1000000);

	this->dimBlock.x = THREADS_PER_BLOCK;
	this->dimBlock.y = 1;
	this->dimBlock.z = 1;

	//Grid dimension when having one thread per gene
	if((n*p*K)%THREADS_PER_BLOCK==0)
		this->dimGrid.x = (n*p*K)/THREADS_PER_BLOCK;
	else
		this->dimGrid.x = (n*p*K)/THREADS_PER_BLOCK +1;
	this->dimGrid.y = 1;
	this->dimGrid.z = 1;

	//Grid dimension when having one thread per chromosome
	if((p*K)%THREADS_PER_BLOCK==0)	
		this->dimGridChromo.x = (p*K)/THREADS_PER_BLOCK;
	else
		this->dimGridChromo.x = (p*K)/THREADS_PER_BLOCK +1;
	this->dimGridChromo.y = 1;
	this->dimGridChromo.z = 1;

	//Grid dimension when having one block to process each chromosome
	this->dimGrid_population.x = p*K; //one block to process each chromosome
	this->dimGrid_population.y = 1;
	this->dimGrid_population.z = 1;

	// Create pseudo-random number generator 
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	// Set seed 
	curandSetPseudoRandomGeneratorSeed(gen, RAND_SEED);
	//Initialize population with random alleles with generated random floats on device 
	reset_population();

}

BRKGA::~BRKGA(){
	// Cleanup 
	curandDestroyGenerator(gen);

	cudaFree(d_population);
	cudaFree(d_population2);
	free(h_population);  

	cudaFree(d_scores);     
	free(h_scores);

	cudaFree(d_scores_idx);
	free(h_scores_idx);

	cudaFree(d_chromosome_gene_idx);
	free(h_chromosome_gene_idx);

	cudaFree(d_random_elite_parent);
	cudaFree(d_random_parent);

	cudaFree(d_best_solutions);
	free(h_best_solutions);


	if(d_instance_info != NULL){
		cudaFree(d_instance_info);
		d_instance_info = NULL;
	}
}


void BRKGA::test_memory_malloc(cudaError_t err, unsigned code, unsigned total_memory){
	if(err != cudaSuccess){
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
		fprintf(stderr, "In cudaMalloc: %u with total memory %u\n", code, total_memory);
		exit(1);
	}
}


/***
	Allocate information used to evaluate chromosomes on the device.
	It also receives the number of elements (num) in the array info and the size (size) of each element.
	Notice we assume the type of the info elements to be float.
***/
void BRKGA::setInstanceInfo(void *info, long unsigned num, long unsigned size){
	if(info != NULL){
		size_t total_memory = num*size;
		printf("Extra Memory Used In GPU due to Instance Info %zu bytes(%zu Mbytes)\n", total_memory, total_memory/1000000);

		if(decode_type != HOST_DECODE){
			test_memory_malloc(cudaMalloc((void **)&d_instance_info, num*size),8,total_memory);
			cudaMemcpy(d_instance_info, info, num*size, cudaMemcpyHostToDevice);
		}
		h_instance_info = info;
	}
}

void BRKGA::setInstanceInfo2D(float *info, long unsigned columns, long unsigned rows, long unsigned size){
	if(info != NULL){
		size_t total_memory = columns*rows*size;
		printf("Extra Memory Used In GPU due to Instance Info %zu bytes(%zu Mbytes)\n", total_memory, total_memory/1000000);

		if(decode_type == DEVICE_DECODE || decode_type == DEVICE_DECODE_CHROMOSOME_SORTED 
					|| decode_type == DEVICE_DECODE_CHROMOSOME_SORTED_COALESCED){
			test_memory_malloc(cudaMalloc((void **)&d_instance_info, columns*rows*size),8,total_memory);
			cudaMemcpy(d_instance_info, info, columns*rows*size, cudaMemcpyHostToDevice);
		}else if(decode_type == DEVICE_DECODE_CHROMOSOME_SORTED_TEXTURE){
	    CUDA_SAFE_CALL(cudaMallocPitch((void**)&d_instance_info,&pitch,columns*sizeof(float),rows));
	    //printf("PITCH %zu -- COlUMNS SIZE %zu\n",pitch,columns*sizeof(float));
	    CUDA_SAFE_CALL(cudaMemcpy2D(d_instance_info, pitch, info, columns*sizeof(float),
                                columns*sizeof(float),rows,cudaMemcpyHostToDevice));
    	tex.normalized = false;
    	size_t tex_ofs;
    	CUDA_SAFE_CALL (cudaBindTexture2D (&tex_ofs, &tex, d_instance_info, &tex.channelDesc,
                                       columns, rows, pitch));
	    if (tex_ofs !=0) {
        printf ("tex_ofs = %zu\n", tex_ofs);
        exit(0);
  	  }
		}
		h_instance_info = info;
	}
}

/***
	Generate random alleles for all chromosomes on GPGPU.
***/
void BRKGA::reset_population(void){
	curandGenerateUniform(gen, d_population, number_chromosomes*chromosome_size);
}


/***
	If HOST_DECODE is used then this function decodes each cromosome with the host_decode function
	provided in Decoder.cpp
***/
void BRKGA::evaluate_chromosomes_host(){
	cudaMemcpy(h_population, d_population, number_chromosomes*chromosome_size*sizeof(float),cudaMemcpyDeviceToHost);

	#pragma omp parallel for default(none) shared(dimGrid,dimBlock,h_population,h_scores) num_threads(OMP_THREADS)
	for(int i=0; i<number_chromosomes; i++){
			float *chromosome = h_population + (i*chromosome_size);
			h_scores[i] = host_decode(chromosome, chromosome_size, h_instance_info);
	}
	cudaMemcpy(d_scores, h_scores, number_chromosomes*sizeof(float),cudaMemcpyHostToDevice);
}



/***
	If DEVICE_DECODE is used then this kernel function decodes each cromosome with the device_decode function
	provided in Decoder.cpp.
	We use one thread per cromosome to process them.
***/
__global__ 
void decode(float *d_scores, float *d_population, unsigned chromosome_size, void *d_instance_info, unsigned number_chromosomes){
	unsigned global_tx = blockIdx.x*blockDim.x + threadIdx.x;	
	if(global_tx < number_chromosomes)
		d_scores[global_tx] = device_decode(d_population + global_tx*chromosome_size, chromosome_size, d_instance_info);
}
/***
	If DEVICE_DECODE is used then this function decodes each cromosome with the kernel function
	decode above.
***/
void BRKGA::evaluate_chromosomes_device(){
	//Make a copy of chromossomes to d_population2 such that they can be messed up inside
	//the decoder functions without afecting the real chromosomes on d_population.
	cudaMemcpy(d_population2, d_population, number_chromosomes*chromosome_size*sizeof(float),cudaMemcpyDeviceToDevice);
	decode<<<dimGridChromo, dimBlock>>>(d_scores, d_population2, chromosome_size, d_instance_info, number_chromosomes);
}

/***
	If DEVICE_DECODE_CHROMOSOME_SORTED is used then this kernel function decodes each cromosome with the
	device_decode_chromosome_sorted function	provided in Decoder.cpp.
	We use one thread per cromosome to process them.

	Notice that we use the struct ChromosomeGeneIdxPair since the cromosome is given already sorted to
	the function, and so it has a field with the original index of each gene in the cromosome.
***/
__global__ 
void decode_chromosomes_sorted(float *d_scores, ChromosomeGeneIdxPair *d_chromosome_gene_idx, int chromosome_size, void *d_instance_info, unsigned number_chromosomes){
	unsigned global_tx = blockIdx.x*blockDim.x + threadIdx.x;	
	if(global_tx < number_chromosomes)
		d_scores[global_tx] = device_decode_chromosome_sorted(d_chromosome_gene_idx + global_tx*chromosome_size, 
																													chromosome_size, d_instance_info);
}
/***
	If DEVICE_DECODE_CHROMOSOME_SORTED is used then this function decodes each cromosome with the kernel function
	decode_chromosomes_sorted above. But first we sort each chromosome by its genes values. We save this information
	in the struct ChromosomeGeneIdxPair d_chromosome_gene_idx.
***/
void BRKGA::evaluate_chromosomes_sorted_device(){
	sort_chromosomes_genes();
	decode_chromosomes_sorted<<<dimGridChromo, dimBlock>>>(d_scores, d_chromosome_gene_idx, chromosome_size,
																											 d_instance_info, number_chromosomes);
}


/***
	If DEVICE_DECODE_CHROMOSOME_SORTED_TEXTURE is used then this kernel function decodes each cromosome with the
	device_decode_chromosome_sorted_texture function	provided in Decoder.cpp.
	We use one thread per cromosome to process them.
***/
__global__ 
void decode_chromosomes_sorted_texture(float *d_scores, ChromosomeGeneIdxPair *d_chromosome_gene_idx, int chromosome_size, 
																										void *d_instance_info, unsigned number_chromosomes){
	unsigned global_tx = blockIdx.x*blockDim.x + threadIdx.x;	
	if(global_tx < number_chromosomes)
		d_scores[global_tx] = device_decode_chromosome_sorted_texture(d_chromosome_gene_idx + global_tx*chromosome_size, 
																													chromosome_size, d_instance_info);
}
/***
	If DEVICE_DECODE_CHROMOSOME_SORTED_TEXTURE is used then this function decodes each cromosome with the kernel function
	decode_chromosomes_sorted_texture above.
***/
void BRKGA::evaluate_chromosomes_sorted_device_texture(){
	sort_chromosomes_genes();
	//here we are supposed to have one thread per chromossome
	//so in the function call dimGridChromo is used
	decode_chromosomes_sorted_texture<<<dimGridChromo, dimBlock>>>(d_scores, d_chromosome_gene_idx, chromosome_size, 
	 	                                                                    d_instance_info, number_chromosomes);
}


/***
	If DEVICE_DECODE_CHROMOSOME_SORTED_COALESCED is used then this function decodes each cromosome with the kernel function
	decode_chromosomes_sorted_texture above.
***/
void BRKGA::evaluate_chromosomes_sorted_device_coalesced(){
	sort_chromosomes_genes();
	//here we are supposed to have one block of threads per chromossome
	//so in the function call dimGrid_population is used
	device_decode_chromosome_sorted_coalesced<<<dimGrid_population, dimBlock>>>(d_chromosome_gene_idx, chromosome_size, 
	 	                                                                    d_instance_info, d_scores);
}


/*
void BRKGA::evaluate_chromosomes_sorted_device2(){
	sort_chromosomes_genes();
	//here we are supposed to have one thread per chromossome
	//so in the function call dimGridChromo is used
	decode_chromosomes_sorted2<<<dimGridChromo, dimBlock>>>(d_scores, d_chromosome_gene_idx, chromosome_size, d_instance_info);


	cudaMemcpy(h_scores, d_scores, number_chromosomes*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_population, d_population, number_chromosomes*chromosome_size*sizeof(float),cudaMemcpyDeviceToHost);
	for(int p=0; p<number_populations; p++){
		printf("\n\nPopulation %d\n", p);
		for(int c=0; c<population_size; c++){
			printf("Chromosome %d - (",c);
			for(int g=0; g<chromosome_size; g++){
				printf("%f,",h_population[c*chromosome_size + g]);
			}
			printf(") Value: %.2f\n", h_scores[c]);
		}
	}
}*/

/***
 If DEVICE_DECODE_CHROMOSOME_SORTED_** is used.
 Kernel function used to save for each gene of each chromosome, the chromosome index, and the original gene index.
 Used later to sort all chromossomes by gene values. We save gene indexes to preserve this information after sorting.
***/
__global__ 
void device_set_chromosome_gene_idx(ChromosomeGeneIdxPair *d_chromosome_gene_idx, unsigned chromosome_size, unsigned number_genes){
	unsigned tx = blockIdx.x*blockDim.x + threadIdx.x;
	if(tx < number_genes){
		unsigned chromosome_idx = tx/chromosome_size;
		unsigned gene_idx = tx%chromosome_size;
		d_chromosome_gene_idx[tx].chromosomeIdx = chromosome_idx;
		d_chromosome_gene_idx[tx].geneIdx = gene_idx;		
	}
}

/***
 If DEVICE_DECODE_CHROMOSOME_SORTED_** is used.
 Used as comparator to sort genes of the chromosomes. 
 After sorting by gene we need to reagroup genes by their chromosomes which are indexed by threadIdx.
***/
__device__ bool operator<(const ChromosomeGeneIdxPair& lhs, const ChromosomeGeneIdxPair& rhs){
	return lhs.chromosomeIdx < rhs.chromosomeIdx;
}

/***
  If DEVICE_DECODE_CHROMOSOME_SORTED_** is used.
	We sort the genes of each chromosome.
	We perform 2 stable_sort sorts: first we sort all genes of all chromosomes by their values, and than we sort by the chromosome index,
	and since stable_sort is used, for each chromosome we will have its genes sorted by their values.
***/
void BRKGA::sort_chromosomes_genes(){
	//First set for each gene, its chromosome index and its original index in the chromosome
	device_set_chromosome_gene_idx<<<dimGrid, dimBlock>>>(d_chromosome_gene_idx, chromosome_size, number_genes);
  //we use d_population2 to sorte all genes by their values
	cudaMemcpy(d_population2, d_population, number_chromosomes*chromosome_size*sizeof(float), cudaMemcpyDeviceToDevice);
	
	thrust::device_ptr<float> keys(d_population2);
	thrust::device_ptr<ChromosomeGeneIdxPair> vals(d_chromosome_gene_idx);

	//stable sort both d_population2 and d_chromosome_gene_idx by all the genes values
	thrust::stable_sort_by_key(keys, keys + number_chromosomes*chromosome_size, vals);
	//stable sort both d_population2 and d_chromosome_gene_idx by the chromosome index values
	thrust::stable_sort_by_key(vals, vals + number_chromosomes*chromosome_size, keys);
}



/***
Kernel function, where each thread process one gene of one chromosome. It receives the current population *d_population, the next population
pointer *d_population2, two random vectors for indices of parents, d_random_elite_parent and d_random_parent,
***/
__global__ 
void device_next_population(float *d_population, float *d_population2, 
	float *d_random_elite_parent, float *d_random_parent, int chromosome_size, 
	unsigned population_size, unsigned elite_size, unsigned mutants_size, float rhoe, PopIdxThreadIdxPair *d_scores_idx,
	unsigned number_genes){

	unsigned tx = blockIdx.x*blockDim.x + threadIdx.x; //global thread index pointing to some gene of some chromosome
	if(tx < number_genes){
		unsigned chromosome_idx = tx/chromosome_size; //global chromosome index having this gene
		unsigned gene_idx = tx%chromosome_size; //the index of this gene in this chromosome

		unsigned pop_idx = chromosome_idx/population_size; //the population index of this chromosome
		unsigned inside_pop_idx = chromosome_idx%population_size; //the chromosome index inside this population

		//if inside_pop_idx < elite_size then the chromosome is elite, so we copy elite gene
		if(inside_pop_idx < elite_size){
			unsigned elite_chromosome_idx = d_scores_idx[chromosome_idx].thIdx; //previous elite chromosome corresponding to this chromosome
			d_population2[tx] = d_population[elite_chromosome_idx*chromosome_size + gene_idx];
		}else if(inside_pop_idx < population_size - mutants_size){
			//thread is responsible to crossover of this gene of this chromosome_idx
			//below are the inside population random indexes of a elite parent and regular parent for crossover
			unsigned inside_parent_elite_idx = (unsigned)(ceilf(d_random_elite_parent[chromosome_idx]*elite_size)-1);
			unsigned inside_parent_idx = (unsigned)(elite_size+ceilf(d_random_parent[chromosome_idx]*(population_size-elite_size))-1);

			unsigned elite_chromosome_idx = d_scores_idx[pop_idx*population_size + inside_parent_elite_idx].thIdx;
			unsigned parent_chromosome_idx = d_scores_idx[pop_idx*population_size + inside_parent_idx].thIdx;
			if(d_population2[tx] <= rhoe)
				//copy allele from elite parent
				d_population2[tx] = d_population[elite_chromosome_idx*chromosome_size + gene_idx];
			else
				//copy allele from regular parent
				d_population2[tx] = d_population[parent_chromosome_idx*chromosome_size + gene_idx];
		}//in the else case the thread corresponds to a mutant and nothing is done.	
	}
}



/***
Main function of the BRKGA algorithm. It evolves K populations for a certain number of generations.
***/
void BRKGA::evolve(int number_generations){
	using std::domain_error;

	if(decode_type == DEVICE_DECODE){
		evaluate_chromosomes_device();
	}else if(decode_type == DEVICE_DECODE_CHROMOSOME_SORTED){
		evaluate_chromosomes_sorted_device();
	}else if(decode_type == DEVICE_DECODE_CHROMOSOME_SORTED_TEXTURE){
		evaluate_chromosomes_sorted_device_texture();
	}else if(decode_type == DEVICE_DECODE_CHROMOSOME_SORTED_COALESCED){
		evaluate_chromosomes_sorted_device_coalesced();
	}else if(decode_type == HOST_DECODE){
		evaluate_chromosomes_host();
	}else{
		throw domain_error("Function decode type is unknown");
	}

	//After this call the vector d_scores_idx has all threads sorted by population, and
	//inside each population, threads are sorted by score
	sort_chromosomes();

	//This call initialize the whole area of the next population d_population2 with random values.
	//So mutantes are already build. For the non mutants we use the 
	//random values generated here to perform the crossover on the current population d_population.
	initialize_population(2);

	//generate random numbers to index parents used for crossover
	curandGenerateUniform(gen, d_random_elite_parent, number_chromosomes);
	curandGenerateUniform(gen, d_random_parent, number_chromosomes);

	//Kernel function, where each thread process one chromosome of the next population.
	device_next_population<<<dimGrid, dimBlock>>>(d_population, d_population2,  d_random_elite_parent,
		d_random_parent, chromosome_size, population_size, elite_size, mutants_size, rhoe, d_scores_idx, number_genes);

	float *aux = d_population2;
	d_population2 = d_population;
	d_population = aux;
}


void BRKGA::initialize_population(int p){
	if(p==1)
		curandGenerateUniform(gen, d_population, number_chromosomes*chromosome_size);
	if(p==2)
		curandGenerateUniform(gen, d_population2, number_chromosomes*chromosome_size);
}



/***
Kernel function that sets for each cromosome its global index (among all populations) and its population index.
***/
__global__ 
void device_set_idx(PopIdxThreadIdxPair *d_scores_idx, unsigned population_size, unsigned number_chromosomes){
	int tx = blockIdx.x*blockDim.x + threadIdx.x;
	if(tx < number_chromosomes){
		d_scores_idx[tx].popIdx = tx/population_size; 
		d_scores_idx[tx].thIdx = tx; 
	}
}

/***
Function used to sort chromosomes by population index
***/
__device__ bool operator<(const PopIdxThreadIdxPair& lhs, const PopIdxThreadIdxPair& rhs){
	return lhs.popIdx < rhs.popIdx;
}

/***
 We sort chromosomes for each population.
 We use the thread index to index each population, and perform 2 stable_sort sorts: first we sort by the chromosome scores, 
 and than by their population index, and since stable_sort is used in each population the chromosomes are sorted by scores.
***/
void BRKGA::sort_chromosomes(){
	//For each chromosome we store in d_scores_idx the global chromosome index and its population index.
	device_set_idx<<<dimGridChromo, dimBlock>>>(d_scores_idx, population_size, number_chromosomes);

	thrust::device_ptr<float> keys(d_scores);
	thrust::device_ptr<PopIdxThreadIdxPair> vals(d_scores_idx);
	//now sort all chromosomes by their scores
	thrust::stable_sort_by_key(keys, keys + number_chromosomes, vals);
	//now sort all chromossomes by their population index
	//in the sorting process it is used operator< above to compare two structs of this type
	thrust::stable_sort_by_key(vals, vals + number_chromosomes, keys);
}



/***
	Kernel function to operate the exchange of elite chromosomes.
	It was launched M*number_populations threads.
	For each population each one of M threads do the copy of an elite chromosome of its own population
	into the other populations.
	To do: make kernel save in local memory the chromosome and then copy to each other population
***/
__global__ 
void device_exchange_elite(float *d_population,  int chromosome_size, unsigned population_size, unsigned number_populations, PopIdxThreadIdxPair *d_scores_idx, unsigned M){

	unsigned tx = threadIdx.x; //this thread value between 0 and M-1
	unsigned pop_idx = blockIdx.x; //this thread population index, a value between 0 and number_populations-1
	unsigned elite_idx = pop_idx*population_size + tx;
	unsigned elite_chromosome_idx = d_scores_idx[elite_idx].thIdx;
	unsigned inside_destiny_idx = population_size-1-(M*pop_idx)-tx;//index of the destiny of this thread inside each population

	for(int i=0; i<number_populations; i++){
		if(i != pop_idx){
			unsigned destiny_chromosome_idx = d_scores_idx[i*population_size + inside_destiny_idx].thIdx;
			for(int j=0; j<chromosome_size;j++)
				d_population[destiny_chromosome_idx*chromosome_size + j] = d_population[elite_chromosome_idx*chromosome_size + j];
		}
	}
}

/***
Exchange M individuals among the different populations.
***/
void BRKGA::exchangeElite(unsigned M){
	using std::range_error;
	if(M > elite_size) { throw range_error("Exchange elite size M greater than elite size."); }
	if(M*number_populations > population_size) { throw range_error("Total exchange elite size greater than population size."); }

	using std::domain_error;
	if(decode_type == DEVICE_DECODE){
		evaluate_chromosomes_device();
	}else if(decode_type == DEVICE_DECODE_CHROMOSOME_SORTED){
		evaluate_chromosomes_sorted_device();
	}else if(decode_type == DEVICE_DECODE_CHROMOSOME_SORTED_TEXTURE){
		evaluate_chromosomes_sorted_device_texture();
	}else if(decode_type == DEVICE_DECODE_CHROMOSOME_SORTED_COALESCED){
		evaluate_chromosomes_sorted_device_coalesced();
	}else if(decode_type == HOST_DECODE){
		evaluate_chromosomes_host();
	}
	else{
		throw domain_error("Function decode type is unknown");
	}

	sort_chromosomes();
	device_exchange_elite<<<number_populations, M>>>(d_population, chromosome_size, population_size, number_populations, d_scores_idx,  M);
}


/***
	Return a vector of vectors, where each line vector corresponds to a chromosome,
	where in position 0 we have its score and in positions 1 to chromosome_size the aleles values
***/
std::vector<std::vector <float>> BRKGA::getkBestChromosomes(unsigned k){
	std::vector<std::vector <float>> ret(k, std::vector<float>(chromosome_size+1));

	global_sort_chromosomes();
	cudaMemcpy(h_scores_idx, d_scores_idx, number_chromosomes*sizeof(PopIdxThreadIdxPair),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_scores, d_scores, number_chromosomes*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_population, d_population, number_chromosomes*chromosome_size*sizeof(float),cudaMemcpyDeviceToHost);

	for(int i=0; i<k; i++){
		unsigned tx = h_scores_idx[i].thIdx;
		float *begin = &h_population[tx*chromosome_size];
		ret[i][0] = h_scores[i];
		for(int u=1; u <= chromosome_size; u++){
			ret[i][u] = begin[u-1];
		}
	}
	return ret;
}

/***
  Return a vector of vectors, where each line vector corresponds to a chromosome,
  where in position 0 we have its score and in positions 1 to chromosome_size the aleles values
***/
std::vector<std::vector <float>> BRKGA::getkBestChromosomes2(unsigned k){
	if(k>POOL_SIZE) k=POOL_SIZE;
	std::vector<std::vector <float>> ret(k, std::vector<float>(chromosome_size+1));
	saveBestChromosomes();
	cudaMemcpy(h_best_solutions, d_best_solutions, POOL_SIZE*(chromosome_size+1)*sizeof(float),cudaMemcpyDeviceToHost);

	for(int i=0; i<k; i++){
		for(int j=0; j <= chromosome_size; j++){
			ret[i][j] = h_best_solutions[i*(chromosome_size+1) + j];
		}
	}

	return ret;
}


__global__ 
void device_save_best_chromosomes(float *d_population,  unsigned chromosome_size,  PopIdxThreadIdxPair *d_scores_idx, float *d_best_solutions, float *d_scores, unsigned best_saved){
	if(!best_saved){//this is the first time saving best solutions in to the pool
		for(int i=0; i<POOL_SIZE; i++){
			unsigned tx = d_scores_idx[i].thIdx;
			float *begin = (float *)&d_population[tx*chromosome_size];
			d_best_solutions[i*(chromosome_size+1)] = d_scores[i]; //save the value of the chromosome
			for(int j=1; j <= chromosome_size; j++){ //save the chromosome
				d_best_solutions[i*(chromosome_size+1)+j] = begin[j-1];
			}
		}
	}else{//Since best solutions were already saved
				//only save now if the i-th best current solution is better than the i-th best overall
		for(int i=0; i<POOL_SIZE; i++){
			unsigned tx = d_scores_idx[i].thIdx;
			float *begin = (float *)&d_population[tx*chromosome_size];
			if(d_scores[i] < d_best_solutions[i*(chromosome_size+1)]){
				d_best_solutions[i*(chromosome_size+1)] = d_scores[i];
				for(int j=1; j <= chromosome_size; j++){
					d_best_solutions[i*(chromosome_size+1)+j] = begin[j-1];
				}
			}
		}
	}
}

/***
 This Function saves in the pool d_best_solutions and h_best_solutions the best solutions generated so far among all populations.
***/
void BRKGA::saveBestChromosomes(){
	global_sort_chromosomes();
	device_save_best_chromosomes<<<1, 1>>>(d_population, chromosome_size, d_scores_idx, d_best_solutions, d_scores, best_saved);
	best_saved = 1;
}

/***
	We sort all chromosomes of all populations toguether.
	We use the global thread index to index each chromosome, since each thread is responsible for one thread.
	Notice that in this function we only perform one sort, since we want the best chromosomes overall, so we do not
	perform a second sort to separate chromosomes by their population.
***/
void BRKGA::global_sort_chromosomes(){
	using std::domain_error;
	if(decode_type == DEVICE_DECODE){
		evaluate_chromosomes_device();
	}else if(decode_type == DEVICE_DECODE_CHROMOSOME_SORTED){
		evaluate_chromosomes_sorted_device();
	}else if(decode_type == DEVICE_DECODE_CHROMOSOME_SORTED_TEXTURE){
		evaluate_chromosomes_sorted_device_texture();
	}else if(decode_type == DEVICE_DECODE_CHROMOSOME_SORTED_COALESCED){
		evaluate_chromosomes_sorted_device_coalesced();
	}else if(decode_type == HOST_DECODE){
		evaluate_chromosomes_host();
	}
	else{
		throw domain_error("Function decode type is unknown");
	}


	device_set_idx<<<dimGridChromo, dimBlock>>>(d_scores_idx, population_size, number_chromosomes);
	thrust::device_ptr<float> keys(d_scores);
	thrust::device_ptr<PopIdxThreadIdxPair> vals(d_scores_idx);
	thrust::sort_by_key(keys, keys + number_chromosomes, vals);
}

