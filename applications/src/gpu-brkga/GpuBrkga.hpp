#ifndef GPUBRKGA_HPP
#define GPUBRKGA_HPP

#include "../common/BrkgaInterface.hpp"

#include <string>
#include <vector>

class GpuBrkga : public BrkgaInterface {
public:
  using BrkgaInterface::Chromosome;
  using BrkgaInterface::Fitness;
  using BrkgaInterface::Population;

  class Decoder {
  public:
    Decoder(unsigned _populationSize,
            unsigned _chromosomeLength,
            unsigned _numberOfThreads,
            bool _isCpuDecode)
        : populationSize(_populationSize),
          chromosomeLength(_chromosomeLength),
          numberOfThreads(_numberOfThreads),
          isCpuDecode(_isCpuDecode) {}

    virtual ~Decoder() = default;

    // Why is it const???
    inline virtual void Init() const {}

    inline virtual void Decode(float* chromosomes, float* fitness) const {
      if (isCpuDecode) {
        DecodeOnCpu(chromosomes, fitness);
      } else {
        DecodeOnGpu(chromosomes, fitness);
      }
    }

  protected:
    // Decode the @p chromosomes using openmp calling @ref DecodeOnCpu below
    virtual void DecodeOnCpu(const float* chromosomes, float* fitness) const;

    virtual Fitness DecodeOnCpu(const float* chromosome) const = 0;

    virtual void DecodeOnGpu(const float* chromosomes,
                             float* fitness) const = 0;

    unsigned populationSize;
    unsigned chromosomeLength;
    unsigned numberOfThreads;
    bool isCpuDecode;
  };

  GpuBrkga(unsigned _chromosomeLength, Decoder* _decoder);
  ~GpuBrkga();

  inline std::string getName() override { return "GPU-BRKGA"; }

  void init(const Parameters& parameters,
            const std::vector<Population>& initialPopulations) override;
  void evolve() override;
  void exchangeElites() override;
  Fitness getBestFitness() override;
  Chromosome getBestChromosome() override;
  std::vector<Population> getPopulations() override;

private:
  class Algorithm;

  Algorithm* algorithm;
  Decoder* decoder;
  Parameters params;
};

#endif  // GPUBRKGA_HPP
