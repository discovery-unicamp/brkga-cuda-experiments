/*
 * TSPInstance.h
 *
 * Reads an instance from TSPLIB (Symmetric TSP).
 *
 * Here's the URL: http://www.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/tsp/
 *
 * Here's the format:
 *
 * NAME : a280
 * COMMENT : drilling problem (Ludwig)
 * TYPE : TSP
 * DIMENSION: 280
 * EDGE_WEIGHT_TYPE : EUC_2D
 * NODE_COORD_SECTION
 * 1 288 149
 * 2 288 129
 * 3 270 133
 * 4 256 141
 * ...
 * EOF
 *
 *  Created on: Mar 16, 2013
 *      Author: Rodrigo
 *
 *      Modified: Eduardo Xavier, 2019
 */

#ifndef TSPINSTANCE_H
#define TSPINSTANCE_H

#include <cmath>
#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <stdio.h>


class TSPInstance {
public:
	typedef std::runtime_error Error;

	TSPInstance(const std::string& instanceFile) throw(Error);
	virtual ~TSPInstance();

	// Getters:
	unsigned getNumNodes() const;
	// Returns the distance from node i to node j:
	double getDistance(unsigned i, unsigned j) const;

private:
	unsigned nNodes;

	class Coord2D {
	public:
		Coord2D() : x(0.0), y(0.0) { }
		Coord2D(double _x, double _y) : x(_x), y(_y) {}

		double getX() const { return x; }
		double getY() const { return y; }

	private:
		double x;
		double y;
	};

	std::vector< Coord2D > nodeCoords;

	int return_dimension(char *s);
	int is_digit(char c);


};

#endif
