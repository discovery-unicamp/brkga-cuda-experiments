#ifndef UTILS_STRINGUTILS_HPP
#define UTILS_STRINGUTILS_HPP 1

#include <string>
#include <vector>

inline bool contains(const std::string& str, const std::string& pattern) {
  return str.find(pattern) != std::string::npos;
}

std::string strip(const std::string& str);

std::vector<std::string> split(const std::string& str, const char delimiter);

#endif  // UTILS_STRINGUTILS_HPP
