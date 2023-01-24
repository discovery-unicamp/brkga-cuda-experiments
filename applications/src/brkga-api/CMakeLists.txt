cmake_minimum_required(VERSION 3.16)
PROJECT(brkga-api-optimizer LANGUAGES CXX)

# Dependencies
find_package(OpenMP REQUIRED)

add_executable(brkga-api
  ${COMMON_SRC}
  ${CMAKE_CURRENT_LIST_DIR}/decoders/CvrpDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/CvrpDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/ScpDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/ScpDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/TspDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/TspDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/BrkgaApi.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BrkgaApi.hpp
  ${CMAKE_CURRENT_LIST_DIR}/BrkgaApiInclude.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
)

target_compile_options(brkga-api PRIVATE "-DUSE_CPP_ONLY")
target_link_libraries(brkga-api PUBLIC OpenMP::OpenMP_CXX PUBLIC -lcurand -fopenmp)

# brkga-api
include_directories(SYSTEM ${BRKGA_API_ROOT})
