/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 */

#ifndef CONFIGFILE_H
#define CONFIGFILE_H

#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>

#define FILE_NAME "config.txt"
#define HOST_DECODE 1
#define DEVICE_DECODE 2
#define DEVICE_DECODE_CHROMOSOME_SORTED 3
constexpr unsigned POOL_SIZE = 10; //size of the pool with best solutions so far


class ConfigFile{
public:
	typedef std::runtime_error Error;

	ConfigFile();
	virtual ~ConfigFile();

	unsigned p; //size of population, exe 256 individuals
	float pe; //proportion of elite population, exe 0.1
	float pm; //proportion of mutant population, exe 0.05
	float rhoe; //probability that child gets an alele from elite parent, exe 0.7
	unsigned K; //number of different populations
	unsigned MAX_GENS;	// run for MAX_GENS generations
	unsigned X_INTVL;	// exchange best individuals at every X_INTVL generations
	unsigned X_NUMBER;	// exchange top X_NUMBER best individuals
	// BRKGA evolution configuration: restart strategy
	unsigned RESET_AFTER; //reset all populations after this number of iterations

	unsigned decode_type; //use decoder on GPU or Host, decoder with sorted aleles or no

};

#endif
