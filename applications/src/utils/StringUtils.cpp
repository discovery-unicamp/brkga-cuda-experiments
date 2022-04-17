#include "StringUtils.hpp"

#include <cctype>
#include <string>
#include <vector>

inline constexpr bool blank(char c) {
  return (isblank(c) || c == '\n' || c == '\r');
}

std::string strip(const std::string& str) {
  if (str.empty()) return str;
  if (!blank(str[0]) && !blank(str.back())) return str;

  std::size_t l = 0;
  std::size_t r = str.size() - 1;
  while (l <= r && blank(str[l])) ++l;
  while (l <= r && blank(str[r])) --r;

  if (l > r) return "";
  return str.substr(l, r - l + 1);
}

std::vector<std::string> split(const std::string& str, const char delimiter) {
  std::vector<std::string> items;
  std::size_t l = 0;
  std::size_t r = 0;
  for (; r < str.size(); ++r) {
    if (str[r] == delimiter) {
      if (l < r) items.push_back(str.substr(l, r - l));
      l = r + 1;
    }
  }
  if (l < r) items.push_back(str.substr(l, r - l));
  return items;
}
