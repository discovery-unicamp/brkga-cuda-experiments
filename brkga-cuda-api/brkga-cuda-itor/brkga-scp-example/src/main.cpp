/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */

#include <stdio.h>
#include <iostream>

#include "BRKGA.h"
#include "ConfigFile.h"
#include "Decoder.h"

int main(int argc, char* argv[]) {
	if(argc < 2) { std::cerr << "usage: <SCPInstance-file>" << std::endl; return -1; }

	const std::string instanceFile = std::string(argv[1]);
	std::cout << "Instance file: " << instanceFile << std::endl;

	// Read the instance:
	SetCoveringDecoder decoder(instanceFile.c_str());	// initialize the instance

	std::cout << argv[0] << ": instance " << argv[1] << " has " << decoder.getNRows()
		<< " rows to be covered with " << decoder.getNColumns() << " columns" << std::endl;

	ConfigFile config;
	BRKGA alg(decoder.getNColumns(), config.p, config.pe, config.pm, config.rhoe, config.K, config.decode_type);

	alg.setInstanceInfo(&decoder, 0, 0);
	
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

	std::vector<std::vector <float>> res2 = alg.getkBestChromosomes2(3);

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
