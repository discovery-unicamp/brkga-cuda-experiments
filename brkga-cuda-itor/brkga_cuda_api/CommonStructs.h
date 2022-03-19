/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */

#ifndef COMMONSTRUCTS_H
#define COMMONSTRUCTS_H

// Given a chromosome, some decoders need to sort it by gene values
// This struct saves for each chromosome the original gene index in that
// chromosome before sorting it.
struct ChromosomeGeneIdxPair {
  unsigned chromosomeIdx;
  unsigned geneIdx;
};

struct ValueIndexPair {
  float first;
  unsigned second;
};

#endif
