#ifndef DECODER_H
#define DECODER_H

#include "BRKGA.h"
#include "CommonStructs.h"

__device__ float device_decode(float *chromosome, int n, void *d_instance_info);
float host_decode(float *chromosome, int n, void *d_instance_info);
__device__ float
device_decode_chromosome_sorted(ChromosomeGeneIdxPair *chromosome, int n,
                                void *d_instance_info);

#endif
