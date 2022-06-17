#ifndef INSTANCES_TSPINSTANCE_HPP
#define INSTANCES_TSPINSTANCE_HPP 1

#include "../Point.hpp"

#include <string>
#include <vector>

class TspInstance {
public:
  static TspInstance fromFile(const std::string& filename);

  TspInstance(TspInstance&& that)
      : numberOfClients(that.numberOfClients),
        distances(std::move(that.distances)) {}

  ~TspInstance() = default;

  [[nodiscard]] inline unsigned chromosomeLength() const {
    return numberOfClients;
  }

  void validate(const float* chromosome, const float fitness) const;

  void validate(const double* chromosome, const double fitness) const;

  void validate(const std::vector<unsigned>& tour, const float fitness) const;

  unsigned numberOfClients;
  std::vector<float> distances;

private:
  TspInstance() : numberOfClients(-1u) {}
};

float getFitness(const unsigned* tour,
                 const unsigned n,
                 const float* distances);

void sortChromosomeToValidate(const float* chromosome,
                              unsigned* permutation,
                              unsigned size);

void sortChromosomeToValidate(const double* chromosome,
                              unsigned* permutation,
                              unsigned size);

#endif  // INSTANCES_TSPINSTANCE_HPP