cmake_minimum_required(VERSION 3.16)
PROJECT(brkga-api-optimizer LANGUAGES CUDA CXX)

# Dependencies
find_package(OpenMP REQUIRED)

# The executable
add_executable(brkga-api
  ${CUDA_COMMON_SRC}
  ${CMAKE_CURRENT_LIST_DIR}/decoders/CvrpDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/CvrpDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/ScpDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/ScpDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/TspDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/TspDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
)

target_link_libraries(brkga-api PUBLIC OpenMP::OpenMP_CXX PUBLIC -lcurand -fopenmp)

# brkga-api
include_directories(SYSTEM ${BRKGA_API_ROOT})

add_library(brkga-api-lib
  ${BRKGA_API_ROOT}/brkgaAPI/BRKGA.h
  ${BRKGA_API_ROOT}/brkgaAPI/MTRand.h
  ${BRKGA_API_ROOT}/brkgaAPI/Population.cpp
  ${BRKGA_API_ROOT}/brkgaAPI/Population.h
)

target_link_libraries(brkga-api LINK_PUBLIC brkga-api-lib)
