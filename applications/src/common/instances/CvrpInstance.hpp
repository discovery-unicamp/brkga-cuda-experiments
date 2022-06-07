#ifndef INSTANCES_CVRPINSTANCE_HPP
#define INSTANCES_CVRPINSTANCE_HPP 1

#include "../Point.hpp"

#include <functional>
#include <string>
#include <vector>

class CvrpInstance {
public:
  static CvrpInstance fromFile(const std::string& filename);

  CvrpInstance(CvrpInstance&& that)
      : capacity(that.capacity),
        numberOfClients(that.numberOfClients),
        distances(std::move(that.distances)),
        demands(std::move(that.demands)) {}

  [[nodiscard]] inline unsigned chromosomeLength() const {
    return numberOfClients;
  }

  void validate(const float* chromosome, const float fitness) const;

  void validate(const double* chromosome, const double fitness) const;

  void validate(const std::vector<unsigned>& tour, const float fitness) const;

  unsigned capacity;
  unsigned numberOfClients;
  std::vector<float> distances;
  std::vector<unsigned> demands;

private:
  CvrpInstance()
      : capacity(static_cast<unsigned>(-1)),
        numberOfClients(static_cast<unsigned>(-1)) {}
};

float getFitness(const unsigned* tour,
                 const unsigned n,
                 const unsigned capacity,
                 const unsigned* demands,
                 const float* distances);

#endif  // INSTANCES_CVRPINSTANCE_HPP
