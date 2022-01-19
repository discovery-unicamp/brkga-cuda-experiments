#include "Point.hpp"

#include <cmath>

float Point::distance(const Point& other) const {
  return std::round(std::hypotf(x - other.x, y - other.y));
}
