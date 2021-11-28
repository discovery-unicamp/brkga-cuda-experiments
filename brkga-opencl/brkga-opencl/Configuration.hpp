#ifndef SRC_CONFIGURATION_HPP
#define SRC_CONFIGURATION_HPP

#include <cassert>
#include <fstream>
#include <istream>
#include <map>
#include "IO.hpp"

class Configuration {
public:

  [[nodiscard]]
  static Configuration fromFile(const std::string& filename);

  [[nodiscard]]
  inline const auto& get(const std::string& param) const {
    assert(values.count(param));
    return values.find(param)->second;
  }

  [[nodiscard]]
  inline int geti(const std::string& param) const {
    return std::stoi(get(param));
  }

  [[nodiscard]]
  inline float getf(const std::string& param) const {
    return std::stof(get(param));
  }

private:

  Configuration() = default;

  std::map<std::string, std::string> values;
};

#endif //SRC_CONFIGURATION_HPP
