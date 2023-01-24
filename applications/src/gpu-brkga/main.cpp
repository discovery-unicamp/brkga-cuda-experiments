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
#include "GpuBrkga.hpp"

class GpuBrkgaRunner : public BrkgaRunner {
public:
  GpuBrkgaRunner(int argc, char** argv)
      : BrkgaRunner(argc, argv), decoder(&instance, params) {
    box::logger::debug("GpuBrkgaRunner was built");
  }

  BrkgaInterface* getBrkga() override {
    return new GpuBrkga(instance.chromosomeLength(), &decoder);
  }

private:
  Decoder decoder;
};

int main(int argc, char** argv) {
  BrkgaRunner::showParams(argc, argv);
  GpuBrkgaRunner(argc, argv).run();
  return 0;
}
