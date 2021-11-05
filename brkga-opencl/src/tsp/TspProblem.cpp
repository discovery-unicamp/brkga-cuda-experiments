// Copyright (c) 2021 Bruno Almêda de Oliveira. All rights reserved.
// Released under the GNU General Public License version 2 or later.

#include "TspProblem.hpp"

TspProblem TspProblem::fromFile(const std::string& filename) {
  info("Reading instance from", filename);
  std::ifstream input(filename);
  TspProblem problem;
  assert(input);  // should be opened

  std::string line;
  while (input >> line && line != "NODE_COORD_SECTION");
  assert(input);  // should not fail

  int id;
  Point p;
  while (input >> id >> p.x >> p.y)
    problem.clients.push_back(p);

  input.close();

  info("The instance has", problem.clients.size(), "clients");
  assert(!problem.clients.empty());
  return problem;
}

float TspProblem::evaluateIndices(const int* chromosome) const {
  const int n = chromosomeLength();

  float fitness = distance(chromosome[n - 1], chromosome[0]);
  for (int i = 1; i < n; ++i)
    fitness += distance(chromosome[i - 1], chromosome[i]);
  assert(fitness > 0);

  return fitness;
}
