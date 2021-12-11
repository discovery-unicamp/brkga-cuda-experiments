/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */

#include <stdio.h>
#include <iostream>
#include <unistd.h>

#include "BRKGA.h"
#include "ConfigFile.h"
#include "SetCoveringDecoder.h"


int main(int argc, char* argv[]) {
	char *par_file = NULL, *inst_file = NULL;
  bool evolve_coalesced = false, evolve_pipeline = false;
  int option;
  unsigned num_pop_pipe = 0, rand_seed = 0;

  while ((option = getopt(argc, argv, "p:i:l:r:c")) !=
         -1) { // get option from the getopt() method
    switch (option) {
    case 'p':
      if (optarg == NULL) {
        std::cout << "No config file with parameters supplied: -p configfile"
                  << std::endl;
        return 0;
      }
      par_file = optarg;
      break;
    case 'i':
      if (optarg == NULL) {
        std::cout << "No instance file supplied: -i instance" << std::endl;
        return 0;
      }
      inst_file = optarg;
      break;
    case 'c':
      evolve_coalesced = true;
      break;
    case 'l':
      evolve_pipeline = true;
      if (optarg == NULL)
        num_pop_pipe = 0;
      else {
        std::cout << optarg << std::endl;
        num_pop_pipe = std::stoi(optarg);
      }
      break;
    case 'r':
      if (optarg == NULL)
        rand_seed = 1;
      else
        rand_seed = std::stoi(optarg);
    }
  }
  if (par_file == NULL || inst_file == NULL) {
    std::cout << "Usage: -p configfile -i instance_file" << std::endl;
    std::cout << "Optional: -c Use coalesced memory." << std::endl;
    std::cout
        << "Optional: -l n Use pipeline with n populations decoded on GPU."
        << std::endl;
    std::cout << "Optional: -r s Sets s as the random seed." << std::endl;
    return 0;
  }

	const std::string instanceFile = std::string(inst_file);
	std::cout << "Instance file: " << instanceFile << std::endl;

	// Read the instance:
	SetCoveringDecoder decoder(instanceFile.c_str());	// initialize the instance

	std::cout << argv[0] << ": instance " << argv[1] << " has " << decoder.getNRows()
		<< " rows to be covered with " << decoder.getNColumns() << " columns" << std::endl;

	BrkgaConfiguration config(par_file);
	//population size is 10*n to use the same rule of Toso and Resende
	config.p = 10 * decoder.getNRows();
	BRKGA alg(&decoder, config, evolve_coalesced, evolve_pipeline, num_pop_pipe, rand_seed);

	for(int i=1; i<=config.MAX_GENS; i++){
		alg.evolve();
		std::cout <<"Evolution: "<< i <<std::endl;
		if(i%config.X_INTVL==0){
			std::cout << "Exchanged top "<< config.X_NUMBER << " best individuals!" << std::endl;
			alg.exchangeElite(config.X_NUMBER);
		}
		if(i%config.RESET_AFTER==0){
			std::cout << "All populations reseted!" << std::endl;
			alg.saveBestChromosomes();
			alg.reset_population();
		}
		//std::vector<std::vector <float>> res = alg.getkBestChromosomes(1);
		//std::cout<<"Value of cuda score: " << res[0][0] << std::endl;
	}

	std::vector<std::vector <float>> res2 = alg.getBestChromosomes(3);

	std::vector<float> aux;
	//aux will be the vector with best solution
	for(int i=1; i<res2[0].size(); i++){
		aux.push_back(res2[0][i]);
	}
	printf("\n");
	printf("Value of best solution: %.2f\n",res2[0][0]);

	//std::vector< float > bestChromosome(aux);
	SetCoveringSolution best(aux, true, true, false, 0.5);
	if(! decoder.verify(best.getSelectedColumns())) {
		std::cerr << "WARNING: Best solution could NOT be verified!" << std::endl;
	}

	std::cout << "\nBest solution:";
	unsigned counter = 0;
	for(unsigned j = 0; j < best.getSelectedColumns().size(); ++j) {
		bool val = best.getSelectedColumns()[j];
			if(val) { std::cout << " " << j; ++counter; }
	}
	std::cout << std::endl;
}
