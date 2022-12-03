#ifndef USE_CPP_ONLY
#error Flag USE_CPP_ONLY must be set
#endif  // USE_CPP_ONLY

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
#include "BrkgaMPIpr.hpp"

class BrkgaMPIprRunner : public BrkgaRunner {
public:
  // TODO add option to set import/export flags

  BrkgaMPIprRunner(int argc, char** argv)
      : BrkgaRunner(argc, argv), decoder(&instance) {}

  BrkgaInterface* getBrkga() override {
    return new BrkgaMPIpr(instance.chromosomeLength(), &decoder);
  }

private:
  Decoder decoder;
};

int main(int argc, char** argv) {
  BrkgaRunner::showParams(argc, argv);
  BrkgaMPIprRunner(argc, argv).run();
  return 0;
}
