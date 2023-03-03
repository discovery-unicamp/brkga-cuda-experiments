#include "Configuration.hpp"

Configuration Configuration::fromFile(const std::string& filename) {
  info("Reading configuration from", filename);
  std::ifstream input(filename);
  Configuration config;
  assert(input);  // file should be opened

  std::string line;
  while (getline(input, line)) {
    if (line.empty()) continue;
    if (line[0] == '#') continue;

    auto index = line.find(':');
    assert(index != std::string::npos);
    assert(line.size() > index + 2);
    assert(line[index + 1] == ' ');

    std::string key = line.substr(0, index);
    std::string value = line.substr(index + 2);

    assert(!key.empty());
    assert(!value.empty());
    assert(key[0] != ' ' && key.back() != ' ');
    assert(value[0] != ' ' && value.back() != ' ');

    config.values[key] = value;
  }
  input.close();

  info("Parameters on configuration:", config.values);
  return config;
}
