cmake_minimum_required(VERSION 3.10)
project(brkga_cuda LANGUAGES CUDA CXX)

set(BB_SEGSORT_ROOT "${CMAKE_CURRENT_LIST_DIR}/bb_segsort")
include("${BB_SEGSORT_ROOT}/CMakeLists.txt")

find_package(OpenMP)
find_package(CUDA REQUIRED)
enable_language(CUDA)
include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}")

add_library(brkga-cuda
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/BRKGA.cu
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/BRKGA.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/BrkgaConfiguration.cpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/BrkgaConfiguration.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/CudaError.cuh
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/CudaUtils.cu
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/CudaUtils.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/DecodeType.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/Logger.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/Decoder.hpp
)

set_target_properties(brkga-cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(brkga-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET brkga-cuda PROPERTY CUDA_STANDARD 11)
set_property(TARGET brkga-cuda PROPERTY CUDA_ARCHITECTURES 61)

target_compile_options(brkga-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp>)
target_link_libraries(brkga-cuda PUBLIC bb_segsort)
target_link_libraries(brkga-cuda PUBLIC OpenMP::OpenMP_CXX PUBLIC -lcurand -fopenmp)
