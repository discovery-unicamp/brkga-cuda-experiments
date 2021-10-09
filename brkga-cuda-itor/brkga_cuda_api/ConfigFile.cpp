/*
 * ConfigFile.cpp
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */

#include "ConfigFile.h"

#include <iostream>
#include <stdio.h>
using namespace std;

ConfigFile::ConfigFile(char *instanceFile) {
  FILE *f = fopen(instanceFile, "r");
  //  FILE *f = fopen(FILE_NAME, "r");

  if (f == NULL) {
    throw Error("ConfigFile: Cannot open config file.");
  }

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
  aux = fscanf(f, "%s %u", st, &decode_type2);
  aux = fscanf(f, "%s %u", st, &OMP_THREADS);
  fclose(f);
}

ConfigFile::~ConfigFile() {}

void ConfigFile::unit_test() {
  cout << "p: " << p << endl;
  cout << "pe: " << pe << endl;
  cout << "pm: " << pm << endl;
  cout << "rhoe: " << rhoe << endl;
  cout << "K: " << K << endl;
  cout << "MAX_GENS: " << MAX_GENS << endl;
  cout << "X_INTVL: " << X_INTVL << endl;
  cout << "X_NUMBER: " << X_NUMBER << endl;
  cout << "RESET_AFTER: " << RESET_AFTER << endl;
  cout << "decode_type: " << decode_type << endl;
  cout << "decode_type2: " << decode_type2 << endl;
  cout << "OMP_THREADS: " << OMP_THREADS << endl;
}
