#include "../Tweaks.hpp"  // Must be generated

#if defined(TSP)
#include "decoders/TspDecoder.hpp"
typedef TspDecoder Decoder;
#elif defined(SCP)
#include "decoders/ScpDecoder.hpp"
typedef ScpDecoder Decoder;
#elif defined(CVRP) || defined(CVRP_GREEDY)
#include "decoders/CvrpDecoder.hpp"
typedef CvrpDecoder Decoder;
#else
#error No known problem defined
#endif

#include "../common/Runner.hpp"
#include "BoxBrkga.hpp"

class BrkgaCuda2Runner : public BrkgaRunner {
public:
  BrkgaCuda2Runner(int argc, char** argv)
      : BrkgaRunner(argc, argv), decoder(&instance) {}

  BrkgaInterface* getBrkga() override {
    return new BoxBrkga(instance.chromosomeLength(), &decoder);
  }

private:
  Decoder decoder;
};

int main(int argc, char** argv) {
  BrkgaRunner::showParams(argc, argv);
  BrkgaCuda2Runner(argc, argv).run();
  return 0;
}
