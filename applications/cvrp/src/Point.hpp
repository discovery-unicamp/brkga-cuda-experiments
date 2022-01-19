#ifndef CVRP_EXAMPLE_SRC_POINT_HPP
#define CVRP_EXAMPLE_SRC_POINT_HPP

struct Point {
  float x, y;
  [[nodiscard]] float distance(const Point& other) const;
};

#endif  // CVRP_EXAMPLE_SRC_POINT_HPP
