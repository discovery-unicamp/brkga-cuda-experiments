#ifndef CVRP_EXAMPLE_SRC_POINT_HPP
#define CVRP_EXAMPLE_SRC_POINT_HPP

#include <cmath>

struct Point {
  float x, y;

  float distance(const Point& other) const { return std::round(std::hypotf(x - other.x, y - other.y)); }
};

#endif  // CVRP_EXAMPLE_SRC_POINT_HPP
