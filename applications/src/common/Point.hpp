#ifndef POINT_HPP
#define POINT_HPP

struct Point {
  float x;
  float y;

  Point() : Point(0, 0) {}

  Point(float _x, float _y) : x(_x), y(_y) {}

  ~Point() = default;

  [[nodiscard]] float distance(const Point& other) const;
};

#endif  // POINT_HPP
