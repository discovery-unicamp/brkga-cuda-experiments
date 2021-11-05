// Copyright (c) 2021 Bruno Almêda de Oliveira. All rights reserved.
// Released under the GNU General Public License version 2 or later.

#ifndef SRC_POINT_HPP
#define SRC_POINT_HPP


#include <cmath>


struct Point {
  [[nodiscard]]
  float distance(const Point& other) const {
    return std::hypotf(x - other.x, y - other.y);
  }

  float x = 0.0;
  float y = 0.0;
};


#endif //SRC_POINT_HPP
