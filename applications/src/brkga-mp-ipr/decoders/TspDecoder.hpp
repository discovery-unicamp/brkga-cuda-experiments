#ifndef TSP_DECODER_HPP
#define TSP_DECODER_HPP

#include <brkga_mp_ipr/fitness_type.hpp>

#include <vector>

class TspInstance;

class TspDecoder {
public:
  TspDecoder(TspInstance* _instance = nullptr) : instance(_instance) {}

  double decode(std::vector<double>& chromosome, bool);

private:
  TspInstance* instance;
};

#endif  // TSP_DECODER_HPP
