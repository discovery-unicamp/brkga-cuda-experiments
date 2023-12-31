cmake_minimum_required(VERSION 3.16)
PROJECT(brkga-mp-ipr-optimizer LANGUAGES CXX)

# Dependencies
find_package(OpenMP REQUIRED)

add_executable(brkga-mp-ipr
  ${COMMON_SRC}
  ${CMAKE_CURRENT_LIST_DIR}/decoders/CvrpDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/CvrpDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/ScpDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/ScpDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/TspDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/TspDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/BrkgaMPIpr.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BrkgaMPIpr.hpp
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
)

target_compile_features(brkga-mp-ipr PRIVATE cxx_std_17)
target_compile_options(brkga-mp-ipr PRIVATE "-DUSE_CPP_ONLY")

# brkga-mp-ipr
include_directories(SYSTEM ${BRKGA_MP_IPR_ROOT})

add_library(brkga-mp-ipr-lib INTERFACE)
target_sources(brkga-mp-ipr-lib INTERFACE
  ${BRKGA_MP_IPR_ROOT}/brkga_mp_ipr/brkga_mp_ipr.hpp
  ${BRKGA_MP_IPR_ROOT}/brkga_mp_ipr/chromosome.hpp
  ${BRKGA_MP_IPR_ROOT}/brkga_mp_ipr/fitness_type.hpp
)
target_compile_options(brkga-mp-ipr-lib INTERFACE "-fopenmp")
target_link_libraries(brkga-mp-ipr-lib INTERFACE OpenMP::OpenMP_CXX)

target_link_libraries(brkga-mp-ipr LINK_PUBLIC brkga-mp-ipr-lib)
