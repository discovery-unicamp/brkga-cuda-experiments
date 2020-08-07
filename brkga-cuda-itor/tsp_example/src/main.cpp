/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */

#include <iostream>
#include <stdio.h>
#include <unistd.h>

#include "BRKGA.h"
#include "ConfigFile.h"
#include "Decoder.h"
#include "TSPInstance.h"

int main(int argc, char *argv[]) {
  int option;
  char *par_file = NULL, *inst_file = NULL;
  bool evolve_coalesced = false;

  while ((option = getopt(argc, argv, ":-p:-i:-c")) !=
         -1) { // get option from the getopt() method
    switch (option) {
    case 'p':
      if (optarg == NULL) {
        printf("No config file with parameters supplied: -p configfile");
        return 0;
      }
      par_file = optarg;
      break;
    case 'i':
      if (optarg == NULL) {
        printf("No instance file supplied: -i instance");
        return 0;
      }
      inst_file = optarg;
      break;
    case 'c':
      evolve_coalesced = true;
      break;
    }
  }
  if (par_file == NULL || inst_file == NULL) {
    std::cout << "Usage: -p configfile -i instance_file" << std::endl;
    return 0;
  }
  const std::string instanceFile = std::string(inst_file);
  std::cout << "Instance file: " << instanceFile << std::endl;
  std::cout << "Use coalesced evolution: " << evolve_coalesced << std::endl;

  // Read the instance:
  TSPInstance instance(instanceFile); // initialize the instance

  long unsigned n = instance.getNumNodes();
  std::cout << "Instance read; here's the info:"
            << "\n\tDimension: " << n << std::endl;

  float *adjMatrix = (float *)malloc(n * n * sizeof(float));
  if (adjMatrix == NULL) {
    std::cout << "Insufficient Memory" << std::endl;
    exit(0);
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      adjMatrix[i * n + j] = instance.getDistance(i, j);
    }
  }

  ConfigFile config(par_file);
  BRKGA alg(n, config);
  alg.setInstanceInfo(adjMatrix, n * n, sizeof(float));
  // alg.setInstanceInfo2D(adjMatrix, n,n, sizeof(float));
  // for(int i=1; i<= 1; i++){
  for (int i = 1; i <= config.MAX_GENS; i++) {
    alg.evolve(evolve_coalesced);
    std::cout << "Evolution: " << i << std::endl;
    if (i % config.X_INTVL == 0) {
      std::cout << "Exchanged top " << config.X_NUMBER << " best individuals!"
                << std::endl;
      alg.exchangeElite(config.X_NUMBER);
    }
    if (i % config.RESET_AFTER == 0) {
      std::cout << "All populations reseted!" << std::endl;
      alg.saveBestChromosomes();
      alg.reset_population();
    }
    // std::vector<std::vector <float>> res = alg.getkBestChromosomes(1);
    // std::cout<<"Value of cuda score: " << res[0][0] << std::endl;
  }

  std::vector<std::vector<float>> res2 = alg.getkBestChromosomes2(3);

  std::vector<double> aux;
  // aux will be the vector with best solution
  for (int i = 1; i < res2[0].size(); i++) {
    aux.push_back(res2[0][i]);
  }
  printf("\n");
  printf("Value of best solution: %.2f\n", res2[0][0]);

  free(adjMatrix);
}
