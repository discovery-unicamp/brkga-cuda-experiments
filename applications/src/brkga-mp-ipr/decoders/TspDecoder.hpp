#ifndef TSP_DECODER_HPP
#define TSP_DECODER_HPP

#include <vector>

class TspInstance;

class TspDecoder {
public:
  TspDecoder(TspInstance* _instance = nullptr) : instance(_instance) {}

  double decode(const std::vector<double>& chromosome, bool) const;

private:
  TspInstance* instance;
};

#endif  // TSP_DECODER_HPP
