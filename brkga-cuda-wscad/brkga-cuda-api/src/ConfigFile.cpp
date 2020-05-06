/*
 * ConfigFile.cpp
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */

#include "ConfigFile.h"

ConfigFile::ConfigFile(){
	FILE *f = fopen(FILE_NAME, "r");

	if(f == NULL) { throw Error("ConfigFile: Cannot open config file."); }

	char st[1000];
	int aux;
	aux = fscanf(f, "%s %u", st, &p);
	aux = fscanf(f, "%s %f", st, &pe);
	aux = fscanf(f, "%s %f", st, &pm);
	aux = fscanf(f, "%s %f", st, &rhoe);
	aux = fscanf(f, "%s %u", st, &K);
	aux = fscanf(f, "%s %u", st, &MAX_GENS);
	aux = fscanf(f, "%s %u", st, &X_INTVL);
	aux = fscanf(f, "%s %u", st, &X_NUMBER);
	aux = fscanf(f, "%s %u", st, &RESET_AFTER);
	aux = fscanf(f, "%s %u", st, &decode_type);
	fclose(f);
}

ConfigFile::~ConfigFile() { }
