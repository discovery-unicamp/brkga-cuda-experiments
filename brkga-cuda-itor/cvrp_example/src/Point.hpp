// Copyright (c) 2021 Bruno Almêda de Oliveira. All rights reserved.
// Released under the GNU General Public License version 2 or later.

#ifndef CVRP_EXAMPLE_SRC_POINT_HPP
#define CVRP_EXAMPLE_SRC_POINT_HPP

#include <cmath>

struct Point {
  float x, y;

  float distance(const Point& other) const {
    return std::round(std::hypotf(x - other.x, y - other.y));
  }
};

#endif //CVRP_EXAMPLE_SRC_POINT_HPP
