#ifndef BRKGA_CUDA_API_DECODE_TYPE_HPP
#define BRKGA_CUDA_API_DECODE_TYPE_HPP

#include <stdexcept>
#include <string>

enum DecodeType {
  /// used to represent an empty variable
  NONE = 0,

  /// decoding is done on CPU (host)
  HOST,

  /// sorts the chromosomes by gene and their index are used to decode on CPU (host)
  HOST_SORTED,

  /// decoding is done no GPU (device)
  DEVICE,

  /// sorts the chromosomes by gene and their index are used to decode on GPU (device)
  DEVICE_SORTED
};

[[nodiscard]] inline std::string getDecodeTypeAsString(DecodeType dt) {
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
  throw std::logic_error("Decode type " + std::to_string(dt) + " wasn't converted to string");
}

#endif  // BRKGA_CUDA_API_DECODE_TYPE_HPP
