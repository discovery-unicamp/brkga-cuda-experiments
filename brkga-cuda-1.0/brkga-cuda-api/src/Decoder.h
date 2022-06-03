#ifndef DECODER_H
#define DECODER_H

#include "BRKGA.h"
#include "CommonStructs.h"

float host_decode(float* results, float *chromosome, int n, void *d_instance_info);
__device__ float device_decode(float* results, float *chromosome, int n, void *d_instance_info);
__device__ float device_decode_chromosome_sorted(float* results, ChromosomeGeneIdxPair *chromosome, int n, void *d_instance_info);

#endif
