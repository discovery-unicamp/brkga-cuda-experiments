cmake_minimum_required(VERSION 3.10)

project(gpu_brkga LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)
include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}")

add_library(gpu-brkga
  ${CMAKE_CURRENT_LIST_DIR}/GPU-BRKGA/cuda_errorchecking.h
  ${CMAKE_CURRENT_LIST_DIR}/GPU-BRKGA/GPUBRKGA.cu
  ${CMAKE_CURRENT_LIST_DIR}/GPU-BRKGA/Individual.h
  ${CMAKE_CURRENT_LIST_DIR}/GPU-BRKGA/kernels.cu
  ${CMAKE_CURRENT_LIST_DIR}/GPU-BRKGA/kernels.h
)

set_property(TARGET gpu-brkga PROPERTY CUDA_STANDARD 17)
set_property(TARGET gpu-brkga PROPERTY CUDA_ARCHITECTURES 61)
set_target_properties(gpu-brkga PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(gpu-brkga PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(gpu-brkga PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp>)
