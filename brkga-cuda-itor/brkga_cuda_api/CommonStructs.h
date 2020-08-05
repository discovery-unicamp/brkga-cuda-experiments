/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */

#ifndef COMMONSTRUCTS_H
#define COMMONSTRUCTS_H

// Used to save for each cuda thread its population index
typedef struct PopIdxThreadIdxPair {
  unsigned popIdx;
  unsigned thIdx;
} PopIdxThreadIdxPair;

// Given a chromossome, some decoders need to sort it by gene values
// This struct saves for each chromosome the original gene index in that
// chromosome before sorting it.
typedef struct ChromosomeGeneIdxPair {
  unsigned chromosomeIdx;
  unsigned geneIdx;
} ChromosomeGeneIdxPair;

typedef struct valueIndexPair {
  float first;
  unsigned second;
} valueIndexPair;

#endif
