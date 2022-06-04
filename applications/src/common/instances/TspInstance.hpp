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

  [[nodiscard]] unsigned chromosomeLength() const { return numberOfClients; }

  void validate(const float* chromosome, const float fitness) const;

  void validate(const std::vector<unsigned>& tour, const float fitness) const;

  unsigned numberOfClients;
  std::vector<float> distances;

private:
  TspInstance() : numberOfClients(-1u) {}
};

#endif  // INSTANCES_TSPINSTANCE_HPP
