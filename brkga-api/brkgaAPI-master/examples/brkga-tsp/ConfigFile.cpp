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
	fscanf(f, "%s %u", st, &p);
	fscanf(f, "%s %f", st, &pe);
	fscanf(f, "%s %f", st, &pm);
	fscanf(f, "%s %f", st, &rhoe);
	fscanf(f, "%s %u", st, &K);
	fscanf(f, "%s %u", st, &MAX_GENS);
	fscanf(f, "%s %u", st, &X_INTVL);
	fscanf(f, "%s %u", st, &X_NUMBER);
	fscanf(f, "%s %u", st, &RESET_AFTER);
	fscanf(f, "%s %u", st, &decode_type);
	fscanf(f, "%s %u", st, &MAXT);
	fclose(f);
}

ConfigFile::~ConfigFile() { }
