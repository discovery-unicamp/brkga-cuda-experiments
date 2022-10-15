CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
project(brkga-applications LANGUAGES CUDA CXX)

# Configuration
SET(CMAKE_CXX_STANDARD 11)

ADD_COMPILE_OPTIONS(
  -Wall -Wextra -Wshadow -Wformat=2 -Wformat-truncation -Wfloat-equal
  -Wshift-overflow -Wconversion -Wno-sign-conversion -Wundef -fno-common
)
ADD_COMPILE_DEFINITIONS(LOG_LEVEL=3)

SET(BRKGA_API_ROOT "${CMAKE_CURRENT_LIST_DIR}/../brkgaAPI")
SET(BRKGA_CUDA_1_ROOT "${CMAKE_CURRENT_LIST_DIR}/../brkga-cuda-1.0")
SET(BRKGA_CUDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../brkga-cuda/src")
SET(BRKGA_MP_IPR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../brkga_mp_ipr_cpp")
SET(GPU_BRKGA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../GPU-BRKGA")

SET(COMMON_SRC
  ${CMAKE_CURRENT_LIST_DIR}/src/common/instances/BaseInstance.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/common/instances/BaseInstance.hpp
  ${CMAKE_CURRENT_LIST_DIR}/src/common/instances/CvrpInstance.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/common/instances/CvrpInstance.hpp
  ${CMAKE_CURRENT_LIST_DIR}/src/common/instances/ScpInstance.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/common/instances/ScpInstance.hpp
  ${CMAKE_CURRENT_LIST_DIR}/src/common/instances/TspInstance.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/common/instances/TspInstance.hpp
  ${CMAKE_CURRENT_LIST_DIR}/src/common/utils/ThrustSort.hpp
  ${CMAKE_CURRENT_LIST_DIR}/src/common/utils/StringUtils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/common/utils/StringUtils.hpp
  ${CMAKE_CURRENT_LIST_DIR}/src/common/Checker.hpp
  ${CMAKE_CURRENT_LIST_DIR}/src/common/MinQueue.hpp
  ${CMAKE_CURRENT_LIST_DIR}/src/common/Parameters.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/common/Parameters.hpp
  ${CMAKE_CURRENT_LIST_DIR}/src/common/Point.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/common/Point.hpp
  ${CMAKE_CURRENT_LIST_DIR}/src/common/Runner.hpp
)

SET(CUDA_COMMON_SRC
  ${COMMON_SRC}
  ${CMAKE_CURRENT_LIST_DIR}/src/common/utils/ThrustSort.cu
  ${CMAKE_CURRENT_LIST_DIR}/src/common/utils/ThrustSort.hpp
)

INCLUDE("${CMAKE_CURRENT_LIST_DIR}/src/brkga-api/CMakeLists.txt")
INCLUDE("${CMAKE_CURRENT_LIST_DIR}/src/brkga-cuda-1.0/CMakeLists.txt")
INCLUDE("${CMAKE_CURRENT_LIST_DIR}/src/brkga-cuda-2.0/CMakeLists.txt")
INCLUDE("${CMAKE_CURRENT_LIST_DIR}/src/brkga-mp-ipr/CMakeLists.txt")
INCLUDE("${CMAKE_CURRENT_LIST_DIR}/src/gpu-brkga/CMakeLists.txt")
