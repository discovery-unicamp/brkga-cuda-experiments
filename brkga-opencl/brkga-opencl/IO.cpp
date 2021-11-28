#include "IO.hpp"

std::mutex printLocker;  // NOLINT
std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
