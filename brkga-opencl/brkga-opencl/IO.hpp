#ifndef SRC_CPP_CONFIG_IO_HPP
#define SRC_CPP_CONFIG_IO_HPP

#include <chrono>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <vector>
#include <map>

template <class T, class U>
std::ostream& operator<<(std::ostream& out, const std::pair<T, U>& p) {
  return out << '(' << p.first << ", " << p.second << ')';
}

template <class T>
auto operator<<(std::ostream& out, const std::vector<T>& v) -> std::ostream& {
  bool flag = false;
  out << '[';
  for (auto& x : v) {
    (flag ? out << ", " : out) << x;
    flag = true;
  }
  return out << ']';
}

template <class T, class U>
std::ostream& operator<<(std::ostream& out, const std::map<T, U>& mp) {
  return out << std::vector(mp.begin(), mp.end());
}

class LoggerOptions {
  bool _flush = false;
  std::string _separator = " ";
  std::string _end = "\n";

public:

  LoggerOptions() = default;
  LoggerOptions(const LoggerOptions&) = delete;
  LoggerOptions(LoggerOptions&&) = delete;
  ~LoggerOptions() = default;

  LoggerOptions& operator=(const LoggerOptions&) = delete;
  LoggerOptions& operator=(LoggerOptions&&) = delete;

  [[nodiscard]]
  auto separator() const -> const std::string& {
    return _separator;
  }

  auto separator(const std::string& s) -> LoggerOptions& {
    _separator = s;
    return *this;
  }

  [[nodiscard]]
  auto end() const -> const std::string& {
    return _end;
  }

  auto end(const std::string& s) -> LoggerOptions& {
    _end = s;
    return *this;
  }

  [[nodiscard]]
  auto flush() const -> bool {
    return _flush;
  }

  auto flush(bool b = true) -> LoggerOptions& {
    _flush = b;
    return *this;
  }
};

const LoggerOptions defaultLoggerOptions;
extern std::mutex printLocker;  // NOLINT
extern std::chrono::steady_clock::time_point startTime;

template <class ...T>
void print(std::ostream& out, const char* file, const int line, const char* func,
           const LoggerOptions& options, const T& ... x) {
  [[maybe_unused]] std::lock_guard<std::mutex> lock(printLocker);

  constexpr int timePrecision = 3;
  constexpr int timeLength = 8 + 1 + timePrecision;  // 8 + dot + precision

  bool flag = false;
  auto separator = [&]() -> std::string {
    if (flag) return options.separator();
    flag = true;
    return "";
  };

  std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
  const auto elapsed = (double)std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count() / 1000.0;
  out << '[' << std::setprecision(timePrecision) << std::fixed << std::setw(timeLength) << elapsed << "s] ";
  out << file << ':' << line << ": on " << func << ": ";
  ((out << separator() << x), ...) << options.end();

  if (options.flush())
    out << std::flush;
}

template <class ...T>
inline void print(std::ostream& out, const char* file, int line, const char* func, const T&... x) {
  print(out, file, line, func, defaultLoggerOptions, x...);
}

#ifndef LOG_LEVEL
#error LOG_LEVEL not defined
#endif // LOG_LEVEL

// can't move those macros to a function since the method called is necessary

#define NO_OPERATION void(nullptr)  // NOLINT

#define printLog(...) print(std::clog, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)  // NOLINT
#define printError(...) print(std::cerr, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)  // NOLINT
#define warn(...) print(std::cerr, __FILE__, __LINE__, __FUNCTION__, "WARNING:", __VA_ARGS__)  // NOLINT

#define error(...) printError(__VA_ARGS__)  // NOLINT
#define info(...) (LOG_LEVEL >= 1 ? printLog(__VA_ARGS__) : NO_OPERATION)  // NOLINT
#define debug(...) (LOG_LEVEL >= 2 ? printLog(__VA_ARGS__) : NO_OPERATION)  // NOLINT
#define trace(...) (LOG_LEVEL >= 3 ? printLog(__VA_ARGS__) : NO_OPERATION)  // NOLINT

#endif //SRC_CPP_CONFIG_IO_HPP
