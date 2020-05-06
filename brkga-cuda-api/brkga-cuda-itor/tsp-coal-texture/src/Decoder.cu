/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */



#include "Decoder.h"

extern texture<float, 2, cudaReadModeElementType> tex;


/***
  Used by host_decode
***/
bool comparator(const valueIndexPair& l, const valueIndexPair& r){ return l.first < r.first; }


/***
	Implement this function if you want to decode cromossomes on the host.
  Parameters are chromosome pointer, its size n, and instance information used to decode.
***/
float host_decode(float *chromosome, int n, void *instance_info){
	float score=0;
	valueIndexPair *valInd = (valueIndexPair *) malloc(n*sizeof(valueIndexPair));
	if(valInd == NULL)
		printf("\nMemory error in host_decode!\n");
	for(int i=0; i<n; i++){
		valInd[i].first = chromosome[i]; //order by the value of genes
		valInd[i].second = i; //original index of gene
	}

	std::sort(valInd, valInd + n, comparator);

	score = 0;
	float *adjMatrix = (float *)instance_info;
	for(int i=0; i<n-1; i++)
		score = score + adjMatrix[valInd[i].second*n + valInd[i+1].second];
	score = score + adjMatrix[valInd[0].second*n + valInd[n-1].second];
	
	free(valInd);
	return score;
	
}


/***
	Implement this function if you want to decode cromossomes on the device in such a way that you will receive a chromosome
	with its genes already sorted in increase order by their values. The struct ChromosomeGeneIdxPair contains the genes
	sorted with their original index in the chromosome saved in geneIdx.
  Parameters are chromosome pointer, its size n, and instance information used to decode.
***/
__device__ float device_decode_chromosome_sorted(ChromosomeGeneIdxPair *chromosome, int n, void *d_instance_info){
	float score = 0;
	float *adjMatrix = (float *)d_instance_info;
	for(int i=0; i<n-1; i++){
		//if(chromosome[i].geneIdx < n  &&  chromosome[i+1].geneIdx <n)
		 score = score + adjMatrix[chromosome[i].geneIdx*n + chromosome[i+1].geneIdx];
	}
	score = score + adjMatrix[chromosome[0].geneIdx*n + chromosome[n-1].geneIdx];
	
	return score;

}


/***
	Identical to the above except that texture memory is used. If there is some type of locality access on d_instance_info
	then this version may be faster than the device_decode_chromosome_sorted.

	Implement this function if you want to decode cromossomes on the device in such a way that you will receive a chromosome
	with its genes already sorted in increase order by their values. The struct ChromosomeGeneIdxPair contains the genes
	sorted with their original index in the chromosome saved in geneIdx.
  Parameters are chromosome pointer, its size n, and instance information used to decode.
***/
__device__ float device_decode_chromosome_sorted_texture(ChromosomeGeneIdxPair *chromosome, int n, void *d_instance_info){
	float score = 0;
	unsigned row, col;
	for(int i=0; i<n-1; i++){
		row = chromosome[i].geneIdx;
		col = chromosome[i+1].geneIdx;
		score = score + tex2D (tex, col, row);
	}
	row = chromosome[0].geneIdx;
	col = chromosome[n-1].geneIdx;
	score = score + tex2D (tex, col, row);

	return score;	
	//return floor(score);

}


/***
	Implement this function if you want to decode cromossomes on the device.
  Parameters are chromosome pointer, its size n, and instance information used to decode.
***/
__device__ float device_decode(float *chromosome, int n, void *d_instance_info){
	valueIndexPair *valInd = (valueIndexPair *) malloc(n*sizeof(valueIndexPair));
	if(valInd == NULL){
		printf("\nMemory error: could not alloc memory in device_decode!\n");
		return 0;
	}
	for(int i=0; i<n; i++){
		valInd[i].first = chromosome[i];
		valInd[i].second = i;
	}
	
	insertionSort(valInd, n);
	//sorting with thrust on device only work with small instances
	//otherwise there are memory allocation problems
	//thrust::device_ptr<valueIndexPair> vals(valInd);
	//thrust::sort(thrust::device, vals, vals+n, comparator2);

	float score = 0;
	float *adjMatrix = (float *)d_instance_info;
	for(int i=0; i<n-1; i++)
		score = score + adjMatrix[valInd[i].second*n + valInd[i+1].second];
	score = score + adjMatrix[valInd[0].second*n + valInd[n-1].second];
	
	free(valInd);
	return score;

}

//Used with thrust sort in the device
__device__ bool comparator2(const valueIndexPair& lhs, const valueIndexPair& rhs){
	return lhs.first < rhs.first;
}

/**
 Just for the purpose of an examplo of using device_decode. This version is slow.
**/
__device__ void insertionSort(valueIndexPair *arr, int n){
   int i, j;
   valueIndexPair key; 
   for (i = 1; i < n; i++) { 
       key = arr[i]; 
       j = i-1; 
       while (j >= 0 && arr[j].first > key.first){ 
           arr[j+1] = arr[j]; 
           j = j-1; 
       } 
       arr[j+1] = key; 
   } 
}





/***
	In this function a block of threads process each cromossome. So the user needs to take care of which
	cromossome is being decoded (given by blockIds.x, since gridSize is equal to the total number of cromossomes).
  The struct ChromosomeGeneIdxPair contains the genes sorted with their original index in the chromosome saved in geneIdx.
***/
__global__ void device_decode_chromosome_sorted_coalesced(ChromosomeGeneIdxPair *chromosomes, int n, void *d_instance_info, float *d_scores){
	unsigned tx = threadIdx.x;
	float *adjMatrix = (float *)d_instance_info;
	ChromosomeGeneIdxPair *chromosome = chromosomes + blockIdx.x*n; //pointer to begnning of the chromosome this thread works on

	//All threads in the block work toguether to decode this chromossome
	__shared__ float sm[THREADS_PER_BLOCK];
	int total;//number of segments in the chromossome to be worked by the threads in this block
	if(n%THREADS_PER_BLOCK == 0)
		total = n/THREADS_PER_BLOCK;
	else
		total = n/THREADS_PER_BLOCK +1;
	sm[tx] = 0;
	__syncthreads();

	
	for(int i=0; i<total; i++){
		unsigned id = i*THREADS_PER_BLOCK + tx;
		if(id + 1 < n){
			unsigned c1 = chromosome[id].geneIdx;
			unsigned c2 = chromosome[id + 1].geneIdx;
			sm[tx] += adjMatrix[c1*n + c2];
		}
	}
	if( (n%THREADS_PER_BLOCK) - 1 == tx) //last id of this tx is id==n-1
			sm[tx] += adjMatrix[chromosome[n-1].geneIdx*n + chromosome[0].geneIdx];
	__syncthreads();

	//do reduction sum of shared memory
	for (unsigned s=THREADS_PER_BLOCK/2; s>0; s>>=1){
		if (tx < s) {
			sm[tx] += sm[tx + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tx == 0){
		d_scores[blockIdx.x] = sm[0];
	} 
}


/*
 The functions below were writen to test the efficiency of decoding on the device x host
 if the decoding functions are cheap (in this case are linear time functions)
 in the size of the chromosome.
*/
float host_decode2(float *chromosome, int n, void *instance_info){
	float aux=0;
	for(int i=0;i<n;i++){
		aux+=chromosome[i];
	}
	float *adjMatrix = (float *)instance_info;
	int i = (int) aux*n;
	i = i%n;
	//returns the distante between city 0 and i \in [0,..,n-1]
	return adjMatrix[0 + i];
}



__device__ float device_decode2(float *chromosome, int n, void *d_instance_info){
	float aux=0;
	for(int i=0;i<n;i++){
		aux+=chromosome[i];
	}
	float *adjMatrix = (float *)d_instance_info;
	int i = (int) aux*n;
	i = i%n;
	return adjMatrix[0 + i];	
}


