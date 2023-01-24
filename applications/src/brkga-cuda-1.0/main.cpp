#ifdef USE_CPP_ONLY
#error Flag USE_CPP_ONLY must be unset
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
#include "BrkgaCuda.hpp"

class BrkgaCudaRunner : public BrkgaRunner {
public:
  BrkgaCudaRunner(int argc, char** argv)
      : BrkgaRunner(argc, argv), decoder(&instance) {
    box::logger::debug("BrkgaCudaRunner was built");
  }

  BrkgaInterface* getBrkga() override {
    return new BrkgaCuda(instance.chromosomeLength(), &decoder);
  }

private:
  Decoder decoder;
};

int main(int argc, char** argv) {
  BrkgaRunner::showParams(argc, argv);
  BrkgaCudaRunner(argc, argv).run();
  return 0;
}
