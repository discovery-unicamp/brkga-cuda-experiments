#include "../Tweaks.hpp"  // Must be generated

#if defined(TSP)
#include "../common/instances/TspInstance.hpp"
#include "decoders/TspDecoder.hpp"
typedef TspInstance Instance;
typedef TspDecoder Decoder;
#elif defined(SCP)
#include "../common/instances/ScpInstance.hpp"
#include "decoders/ScpDecoder.hpp"
typedef ScpInstance Instance;
typedef ScpDecoder Decoder;
#elif defined(CVRP) || defined(CVRP_GREEDY)
#include "../common/instances/CvrpInstance.hpp"
#include "decoders/CvrpDecoder.hpp"
typedef CvrpInstance Instance;
typedef CvrpDecoder Decoder;
#else
#error No problem/instance/decoder defined
#endif

#include "../common/Runner.hpp"
#include "BoxBrkga.hpp"

class BrkgaCuda2Runner : public RunnerBase<Decoder::Fitness, Instance> {
public:
  // TODO add option to set import/export flags
  BrkgaCuda2Runner(int argc, char** argv)
      : RunnerBase(argc, argv), decoder(&instance) {}

  BrkgaInterface* getBrkga() override {
    return new BoxBrkga(instance.chromosomeLength(), &decoder);
  }

private:
  Decoder decoder;
};

int main(int argc, char** argv) {
  RunnerBase<Decoder::Fitness, Instance>::showParams(argc, argv);
  BrkgaCuda2Runner(argc, argv).run();
  return 0;
}
