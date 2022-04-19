#ifndef INSTANCES_CVRPINSTANCE_HPP
#define INSTANCES_CVRPINSTANCE_HPP 1

#include "../Point.hpp"
#include "Instance.hpp"

#include <functional>
#include <string>
#include <vector>

extern unsigned threadsPerBlock;  // FIXME remove this

class CvrpInstance : public Instance {
public:  // decoders
  void hostDecode(unsigned int numberOfChromosomes,
                  const float* chromosomes,
                  float* results) const override;

  void deviceDecode(cudaStream_t stream,
                    unsigned numberOfChromosomes,
                    const float* dChromosomes,
                    float* dResults) const override;

  void hostSortedDecode(unsigned numberOfChromosomes,
                        const unsigned* indices,
                        float* results) const override;

  void deviceSortedDecode(cudaStream_t stream,
                          unsigned numberOfChromosomes,
                          const unsigned* dIndices,
                          float* dResults) const override;

private:
  float getFitness(const unsigned* tour, bool hasDepot) const;

public:  // general
  static CvrpInstance fromFile(const std::string& filename);

  static std::pair<float, std::vector<unsigned> > readBestKnownSolution(
      const std::string& filename);

  CvrpInstance(CvrpInstance&& that)
      : capacity(that.capacity),
        numberOfClients(that.numberOfClients),
        dDistances(that.dDistances),
        dDemands(that.dDemands),
        distances(std::move(that.distances)),
        demands(std::move(that.demands)) {}

  ~CvrpInstance();

  [[nodiscard]] inline unsigned chromosomeLength() const {
    return numberOfClients;
  }

  void validateSortedChromosome(const unsigned* sortedChromosome,
                                const float fitness) const override;

  void validateTour(const std::vector<unsigned>& tour,
                    const float fitness,
                    const bool hasDepot = false) const;

private:
  CvrpInstance()
      : capacity(static_cast<unsigned>(-1)),
        numberOfClients(static_cast<unsigned>(-1)),
        dDistances(nullptr),
        dDemands(nullptr) {}

  unsigned capacity;
  unsigned numberOfClients;
  float* dDistances;
  unsigned* dDemands;
  std::vector<float> distances;
  std::vector<unsigned> demands;
};

#endif  // INSTANCES_CVRPINSTANCE_HPP
