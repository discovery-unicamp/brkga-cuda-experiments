#ifndef APPLICATIONS_CHECKER_HPP
#define APPLICATIONS_CHECKER_HPP

#include <cstdio>
#include <string>

inline void _check_fail(const std::string& condition,
                 const std::string& file,
                 int line,
                 const std::string& func,
                 const std::string& message) {
  std::string log = "Validation `" + condition + "` failed\n";
  log += file + ":" + std::to_string(line) + ": on " + func + ": " + message;
  throw std::logic_error(log);
}

#endif  // APPLICATIONS_CHECKER_HPP`

#undef check
#define check(cond, ...)                                                \
  do {                                                                  \
    if (!static_cast<bool>(cond)) {                                     \
      std::string buf(2048, '.');                                       \
      snprintf((char*)buf.data(), buf.size(), __VA_ARGS__);             \
      _check_fail(#cond, __FILE__, __LINE__, __PRETTY_FUNCTION__, buf); \
    }                                                                   \
  } while (false)
