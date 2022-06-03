#ifndef BRKGA_CUDA_DECODE_TYPE_HPP
#define BRKGA_CUDA_DECODE_TYPE_HPP

#include <stdexcept>
#include <string>

namespace box {
enum DecodeType {
  /// Used to represent an empty variable
  NONE = 0,

  /// Decode on CPU
  HOST,

  /// Decode on CPU the indices sorted using the genes as keys
  HOST_SORTED,

  /// Decode on GPU
  DEVICE,

  /// Decode on GPU the indices sorted using the genes as keys
  DEVICE_SORTED
};

[[nodiscard]] inline DecodeType fromString(const std::string& str) {
  if (str == "none") {
    return NONE;
  } else if (str == "host") {
    return HOST;
  } else if (str == "host-sorted") {
    return HOST_SORTED;
  } else if (str == "device") {
    return DEVICE;
  } else if (str == "device-sorted") {
    return DEVICE_SORTED;
  } else {
    throw std::runtime_error("Unknown decode type: " + str);
  }
}

[[nodiscard]] inline std::string toString(DecodeType dt) {
  switch (dt) {
    case NONE:
      return "none";
    case HOST:
      return "host";
    case HOST_SORTED:
      return "host-sorted";
    case DEVICE:
      return "device";
    case DEVICE_SORTED:
      return "device-sorted";
  }
  throw std::logic_error("Decode type " + std::to_string(dt)
                         + " wasn't converted to string");
}
}  // namespace box

#endif  // BRKGA_CUDA_DECODE_TYPE_HPP
