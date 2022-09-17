#ifdef USE_CPP_ONLY
#error Cannot use thrust with flag USE_CPP_ONLY
#endif  // USE_CPP_ONLY

#ifndef THRUST_SORT_HPP
#define THRUST_SORT_HPP

#include "../../Tweaks.hpp"

void thrustSort(Gene* dKeys, unsigned* dValues, unsigned length);
void thrustSortKernel(Gene* dKeys, unsigned* dValues, unsigned length);

#endif  // THRUST_SORT_HPP
