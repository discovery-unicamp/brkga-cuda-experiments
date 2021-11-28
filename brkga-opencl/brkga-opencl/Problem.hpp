#ifndef SRC_BRKGA_PROBLEM_HPP
#define SRC_BRKGA_PROBLEM_HPP


class Problem {
public:

  [[nodiscard]]
  virtual int chromosomeLength() const = 0;

  [[nodiscard]]
  virtual float evaluateIndices(const int* indices) const = 0;
};


#endif //SRC_BRKGA_PROBLEM_HPP
