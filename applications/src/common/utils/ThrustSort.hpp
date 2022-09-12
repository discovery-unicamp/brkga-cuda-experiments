#ifndef THRUST_SORT_HPP
#define THRUST_SORT_HPP

#ifdef USE_CPP_ONLY
#include <stdexcept>

void thrustSort(float*, unsigned*, unsigned) {
  throw std::logic_error(std::string(__PRETTY_FUNCTION__)
                         + " should not be called");
}

void thrustSort(double*, unsigned*, unsigned) {
  throw std::logic_error(std::string(__PRETTY_FUNCTION__)
                         + " should not be called");
}

void thrustSortKernel(float*, unsigned*, unsigned) {
  throw std::logic_error(std::string(__PRETTY_FUNCTION__)
                         + " should not be called");
}

void thrustSortKernel(double*, unsigned*, unsigned) {
  throw std::logic_error(std::string(__PRETTY_FUNCTION__)
                         + " should not be called");
}
#else
void thrustSort(float* dKeys, unsigned* dValues, unsigned length);
void thrustSort(double* dKeys, unsigned* dValues, unsigned length);
void thrustSortKernel(float* dKeys, unsigned* dValues, unsigned length);
void thrustSortKernel(double* dKeys, unsigned* dValues, unsigned length);
#endif  // USE_CPP_ONLY

#endif  // THRUST_SORT_HPP
