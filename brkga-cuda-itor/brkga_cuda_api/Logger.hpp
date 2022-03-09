#ifndef BRKGA_CUDA_API_LOGGER_HPP
#define BRKGA_CUDA_API_LOGGER_HPP

#ifndef LOG_LEVEL
#define LOG_LEVEL 0
#endif  // LOG_LEVEL

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define RESET "\033[0m"
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m" /* Red */
#define GREEN "\033[32m" /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define BLUE "\033[34m" /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m" /* Cyan */
#define WHITE "\033[37m" /* White */
#define BOLDBLACK "\033[1m\033[30m" /* Bold Black */
#define BOLDRED "\033[1m\033[31m" /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m" /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m" /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m" /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m" /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m" /* Bold White */

template <class T>
std::string str(const std::vector<T>& v, const std::string& sep = ", ") {
  std::stringstream ss;
  bool flag = false;
  ss << '[';
  for (auto& x : v) {
    (flag ? ss << sep : ss) << x;
    flag = true;
  }
  ss << ']';
  return ss.str();
}

template <class... T>
std::string log_nosep(const T&... x) {
  std::stringstream ss;
  ((ss << x), ...);
  return ss.str();
}

template <class... T>
void print(std::ostream& out, const char* file, const int line, const char* func, const char* color, const T&... x) {
  bool flag = false;
  const std::string emptyStr = "";
  const std::string spaceStr = " ";
  auto separator = [&]() {
    if (flag) return spaceStr;
    flag = true;
    return emptyStr;
  };

  out << file << ':' << line << ": on " << func << ": ";
  out << color;
  ((out << separator() << x), ...) << '\n';
  out << RESET;
}

#define NO_OPERATION void(nullptr)  // NOLINT
#define printLog(...) print(std::clog, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)  // NOLINT

#define error(...) (printLog(BOLDRED, __VA_ARGS__))  // NOLINT
#define warning(...) (LOG_LEVEL >= 1 ? printLog(YELLOW, __VA_ARGS__) : NO_OPERATION)  // NOLINT
#define info(...) (LOG_LEVEL >= 2 ? printLog(GREEN, __VA_ARGS__) : NO_OPERATION)  // NOLINT
#define debug(...) (LOG_LEVEL >= 3 ? printLog(CYAN, __VA_ARGS__) : NO_OPERATION)  // NOLINT

#endif  // BRKGA_CUDA_API_LOGGER_HPP
