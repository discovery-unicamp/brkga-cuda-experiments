/*
 * TSPInstance.cpp
 *
 *  Created on: Mar 16, 2013
 *      Author: Rodrigo
 *	Modified: Eduardo Xavier 2019
 */

#include "TSPInstance.h"

TSPInstance::TSPInstance(const std::string& instanceFile) throw(TSPInstance::Error) {

	FILE *f = fopen(instanceFile.c_str(), "r");

	if(f == NULL) { throw Error("TSPInstance: Cannot open input file."); }

	// WARNING: the code below assumes an ordered input file following a280.tsp
	//          it will not work with all instances in the TSPLIB.
	char st[1000];

	fgets(st, 1000, f); //read name line
	fgets(st, 1000, f); //read comment line
	while(st[0]=='C')//read the COMMENTS lines and the type line when leaves the loop
		fgets(st, 1000, f);
	//testFileRead(fgets(st, 1000, f)); //read type line
	fgets(st, 1000, f); //read dimension line
	int dimension = return_dimension(st);
	fgets(st, 1000, f); //read edgen type line
	fgets(st, 1000, f); //read node section line

	int aux; double x,y;

	nNodes = dimension;
 	nodeCoords.reserve(nNodes);

	for(int i=0; i<dimension; i++){
		fscanf(f, "%d %lf %lf", &aux, &x, &y);
		nodeCoords[i] = Coord2D(x, y);
	}

	fclose(f);
}

TSPInstance::~TSPInstance() { }

/*
unsigned TSPInstance::getDistance(unsigned i, unsigned j) const {
	const float x2 = std::pow(nodeCoords[i].getX() - nodeCoords[j].getX(), 2.0);
	const float y2 = std::pow(nodeCoords[i].getY() - nodeCoords[j].getY(), 2.0);

	return std::floor(sqrt(x2 + y2) + 0.5);
}
*/

double TSPInstance::getDistance(unsigned i, unsigned j) const {
	const double x2 = std::pow(nodeCoords[i].getX() - nodeCoords[j].getX(), 2.0);
	const double y2 = std::pow(nodeCoords[i].getY() - nodeCoords[j].getY(), 2.0);


	return std::sqrt(x2 + y2);
}


unsigned TSPInstance::getNumNodes() const { return nNodes; }



int TSPInstance::return_dimension(char *s){
	int i=0;
	while (s[i]!='\0' && !is_digit(s[i]))
		i++;
	int result=0;
	while(is_digit(s[i])){
		result = result*10 + (s[i]-48);
		i++;
	}
	return result;

}

int TSPInstance::is_digit(char c){
	if(c>=48 && c<=57)
		return 1;
	return 0;
}