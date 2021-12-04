/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */

#include "SetCoveringDecoder.h"

bool comparator(const ValueIndexPair& l, const ValueIndexPair& r){ return l.first < r.first; }

/***
	Implement this function if you want to decode cromossomes on the host.
  Parameters are chromosome pointer, its size n, and instance information used to decode.
***/
float host_decode(const float* chromosome, int n, void* instance_info) {
	SetCoveringDecoder* decoder = (SetCoveringDecoder*) instance_info;
	std::vector<float> aux(chromosome, chromosome + n);
	return decoder->decode(aux);
}

void SetCoveringDecoder::evaluateChromosomesOnHost(
		unsigned int numberOfChromosomes,
		const float* chromosomes,
		float* results) const {
	for (unsigned i = 0; i < numberOfChromosomes; ++i) {
		results[i] = host_decode(chromosomes + i * chromosomeLength(), (int) chromosomeLength(), (void*) this);
	}
}
