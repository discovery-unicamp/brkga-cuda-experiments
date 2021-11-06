// Copyright (c) 2021 Bruno Almêda de Oliveira. All rights reserved.
// Released under the GNU General Public License version 2 or later.

#include "IO.hpp"

std::mutex printLocker;  // NOLINT
std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
