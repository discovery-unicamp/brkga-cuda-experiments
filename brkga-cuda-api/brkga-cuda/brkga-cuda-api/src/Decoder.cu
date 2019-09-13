/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */



#include "Decoder.h"

bool comparator(const valueIndexPair& l, const valueIndexPair& r){ return l.first < r.first; }


/***
	Implement this function if you want to decode cromossomes on the host.
  Parameters are chromosome pointer, its size n, and instance information used to decode.
***/
float host_decode(float *chromosome, int n, void *instance_info){
	return 0;	
}

/***
	Implement this function if you want to decode cromossomes on the device in such a way that you will receive a chromosome
	with its genes already sorted in increase order by their values. The struct ChromosomeGeneIdxPair contains the genes
	sorted with their original index in the chromosome saved in geneIdx.
  Parameters are chromosome pointer, its size n, and instance information used to decode.
***/
__device__ float device_decode_chromosome_sorted(ChromosomeGeneIdxPair *chromosome, int n, void *d_instance_info){
	return 0;
}


/***
	Implement this function if you want to decode cromossomes on the device.
  Parameters are chromosome pointer, its size n, and instance information used to decode.
***/
__device__ float device_decode(float *chromosome, int n, void *d_instance_info){
	return 0;
}





