#include "DecodeType.hpp"

#include "Logger.hpp"

#include <cctype>
#include <string>

bool _endsWith(const std::string& str, const std::string& end) {
  return str.length() >= end.length()
         && str.compare(str.length() - end.length(), end.length(), end) == 0;
}

box::DecodeType box::DecodeType::fromString(const std::string& str) {
  box::logger::debug("Parsing decoder:", str);

  bool all = str.substr(0, 3) == "all";
  bool cpu = str.substr((all ? 4 : 0), 3) == "cpu";
  bool chromosome = !_endsWith(str, "permutation");

  auto dt = DecodeType(cpu, chromosome, all);
  if (dt.str() != str)
    throw std::runtime_error("Invalid decoder: " + str + "; did you mean "
                             + dt.str() + "?");

  return dt;
}

std::string box::DecodeType::str() const {
  std::string str = _all ? "all-" : "";
  str += _cpu ? "cpu" : "gpu";
  str += _chromosome ? "" : "-permutation";
  return str;
}
