// Copyright (c) 2021 Bruno Almêda de Oliveira. All rights reserved.
// Released under the GNU General Public License version 2 or later.

#ifndef SRC_TSP_TSPPROBLEM_HPP
#define SRC_TSP_TSPPROBLEM_HPP

#include "Point.hpp"
#include "../brkga/Problem.hpp"
#include "../brkga/IO.hpp"
#include <cassert>
#include <fstream>
#include <istream>
#include <vector>

class TspProblem : public Problem {
public:

  [[nodiscard]]
  static TspProblem fromFile(const std::string& filename);

  [[nodiscard]]
  inline float distance(int u, int v) const {
    return clients[u].distance(clients[v]);
  }

  [[nodiscard]]
  inline int chromosomeLength() const override {
    return (int) clients.size();
  }

  [[nodiscard]]
  float evaluateIndices(const int* chromosome) const override;

private:

  TspProblem() = default;

  std::vector<Point> clients;
};

#endif //SRC_TSP_TSPPROBLEM_HPP
