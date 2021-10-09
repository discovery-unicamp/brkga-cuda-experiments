/*
 * TSPInstance.cpp
 *
 *  Created on: Mar 16, 2013
 *      Author: Rodrigo
 *
 *  Modified: Eduardo Xavier, 2019
 *
 */

#include "TSPInstance.h"

void testFileRead(char *s) {
  if (s == NULL) {
    fprintf(stderr, "Error reading instance file!\n");
    exit(0);
  }
}

TSPInstance::TSPInstance(const std::string &instanceFile) {

  FILE *f = fopen(instanceFile.c_str(), "r");

  if (f == NULL) {
    throw Error("TSPInstance: Cannot open input file.");
  }

  // WARNING: the code below assumes an ordered input file following a280.tsp
  //          it will not work with all instances in the TSPLIB.
  char st[1000];

  testFileRead(fgets(st, 1000, f)); // read name line
  testFileRead(fgets(st, 1000, f)); // read comment line
  while (st[0] ==
         'C') // read the COMMENTS lines and the type line when leaves the loop
    testFileRead(fgets(st, 1000, f));
  // testFileRead(fgets(st, 1000, f)); //read type line
  testFileRead(fgets(st, 1000, f)); // read dimension line
  int dimension = return_dimension(st);
  testFileRead(fgets(st, 1000, f)); // read edgen type line
  testFileRead(fgets(st, 1000, f)); // read node section line

  int aux;
  float x, y;

  nNodes = dimension;
  nodeCoords.reserve(nNodes);
  int r;
  for (int i = 0; i < dimension; i++) {
    r = fscanf(f, "%d %f %f", &aux, &x, &y);
    nodeCoords[i] = Coord2D(x, y);
    // fprintf(stderr, "%d %f %f\n",i,x,y);
  }

  distances = (float *)malloc(nNodes * nNodes * sizeof(float));
  if (distances == NULL) {
    std::cout << "Insufficient Memory" << std::endl;
    exit(0);
  }

  for (unsigned i = 0; i < nNodes; i++) {
    for (unsigned j = 0; j < nNodes; j++) {
      distances[i * nNodes + j] = getDistance(i, j);
    }
  }

  cudaMalloc(&dDistances, nNodes * nNodes * sizeof(float));
  CUDA_CHECK(cudaMemcpy(dDistances, distances, nNodes * nNodes * sizeof(float), cudaMemcpyHostToDevice));

  fclose(f);
}

TSPInstance::~TSPInstance() {
  free(distances);
  CUDA_CHECK(cudaFree(dDistances));
}

/*unsigned TSPInstance::getDistance(unsigned i, unsigned j) const {
        const float x2 = std::pow(nodeCoords[i].getX() -
nodeCoords[j].getX(), 2.0); const float y2 = std::pow(nodeCoords[i].getY() -
nodeCoords[j].getY(), 2.0);

        return std::floor((x2 + y2) + 0.5);
}
*/

unsigned TSPInstance::getNumNodes() const { return nNodes; }

float TSPInstance::getDistance(unsigned i, unsigned j) const {
  return std::hypotf(nodeCoords[i].getX() - nodeCoords[j].getX(), nodeCoords[i].getY() - nodeCoords[j].getY());
}

int TSPInstance::return_dimension(char *s) {
  int i = 0;
  while (s[i] != '\0' && !is_digit(s[i]))
    i++;
  int result = 0;
  while (is_digit(s[i])) {
    result = result * 10 + (s[i] - 48);
    i++;
  }
  return result;
}

int TSPInstance::is_digit(char c) {
  if (c >= 48 && c <= 57)
    return 1;
  return 0;
}
