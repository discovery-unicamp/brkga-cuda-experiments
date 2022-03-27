cmake_minimum_required(VERSION 3.10)
project(gpu_brkga LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)
include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}")

# Makes the library an interface since it only has headers.
add_library(gpu-brkga INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}/GPU-BRKGA/cuda_errorchecking.h
  ${CMAKE_CURRENT_LIST_DIR}/GPU-BRKGA/GPUBRKGA.cuh
  ${CMAKE_CURRENT_LIST_DIR}/GPU-BRKGA/Individual.h
  ${CMAKE_CURRENT_LIST_DIR}/GPU-BRKGA/kernels.cuh
)

set_property(TARGET gpu-brkga PROPERTY CUDA_STANDARD 17)
set_property(TARGET gpu-brkga PROPERTY CUDA_ARCHITECTURES 61)
set_target_properties(gpu-brkga PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(gpu-brkga PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
