// Copyright (c) 2021 Bruno Almêda de Oliveira. All rights reserved.
// Released under the GNU General Public License version 2 or later.

#ifndef CVRP_EXAMPLE_SRC_CVRPINSTANCE_HPP
#define CVRP_EXAMPLE_SRC_CVRPINSTANCE_HPP

#include "Point.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

class CvrpInstance {
public:  // for testing purposes

  static CvrpInstance fromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("File is not open");

    CvrpInstance instance;
    std::string str;

    // read capacity
    instance.capacity = -1;
    while ((file >> str) && str != "NODE_COORD_SECTION") {
      if (str == "CAPACITY") {
        file >> str;  // semicolon
        file >> instance.capacity;
      }
    }

    // read locations
    while ((file >> str) && str != "DEMAND_SECTION") {
      float x, y;
      file >> x >> y;
      instance.locations.push_back({x, y});
    }
    instance.numberOfClients = (int)instance.locations.size() - 1;

    // read demand
    while ((file >> str) && str != "DEPOT_SECTION") {
      int d;
      file >> d;
      instance.demand.push_back(d);
    }

    if (instance.capacity <= 0) throw std::runtime_error("Invalid capacity");
    if (instance.locations.size() <= 1) throw std::runtime_error("Must have locations");
    if (instance.locations.size() != instance.demand.size()) throw std::runtime_error("Missing location or demand");
    if (instance.demand[0] != 0) throw std::runtime_error("Depot with demand");

    const int n = instance.numberOfClients;
    instance.distances.resize((n + 1) * (n + 1));
    for (int i = 0; i <= n; ++i)
      for (int j = i; j <= n; ++j)
        instance.distances[i * (n + 1) + j] = instance.locations[i].distance(instance.locations[j]);

    return instance;
  }

  int capacity;
  int numberOfClients;
  std::vector<float> distances;
  std::vector<Point> locations;
  std::vector<int> demand;
};

#endif //CVRP_EXAMPLE_SRC_CVRPINSTANCE_HPP
