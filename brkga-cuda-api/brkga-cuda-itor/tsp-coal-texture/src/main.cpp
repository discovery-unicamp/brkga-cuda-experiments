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
#include "TSPInstance.h"
#include "ConfigFile.h"
#include "Decoder.h"

int main(int argc, char* argv[]) {
	if(argc < 2) { std::cerr << "usage: <TSPLIB-file>" << std::endl; return -1; }

	const std::string instanceFile = std::string(argv[1]);
	std::cout << "Instance file: " << instanceFile << std::endl;

	// Read the instance:
	TSPInstance instance(instanceFile); 	// initialize the instance

	long unsigned n = instance.getNumNodes();
	std::cout << "Instance read; here's the info:"
			<< "\n\tDimension: " << n << std::endl;


	float *adjMatrix = (float *)malloc(n*n*sizeof(float));
	if(adjMatrix == NULL){
		std::cout << "Insufficient Memory" << std::endl;
		exit(0);
	}

	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			adjMatrix[i*n + j] = instance.getDistance(i,j);
		}
	}

	ConfigFile config;
	BRKGA alg(n, config.p, config.pe, config.pm, config.rhoe, config.K, config.decode_type, config.MAXT);

	//alg.setInstanceInfo(adjMatrix, n*n, sizeof(float));
	alg.setInstanceInfo2D(adjMatrix, n,n, sizeof(float));	
	//for(int i=1; i<= 1; i++){
	for(int i=1; i<= config.MAX_GENS; i++){
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

	std::vector<double> aux;
	//aux will be the vector with best solution
	for(int i=1; i<res2[0].size(); i++){
		aux.push_back(res2[0][i]);
	}
	printf("\n");
	printf("Value of best solution: %.2f\n",res2[0][0]);
	
	free(adjMatrix);
	
}
