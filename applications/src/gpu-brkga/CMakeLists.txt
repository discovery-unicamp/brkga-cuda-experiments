cmake_minimum_required(VERSION 3.16)
PROJECT(gpu-brkga-optimizer LANGUAGES CUDA CXX)

# Dependencies
find_package(CUDA REQUIRED)
enable_language(CUDA)
include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}")

# The executable
add_executable(gpu-brkga
  ${CUDA_COMMON_SRC}
  ${CMAKE_CURRENT_LIST_DIR}/decoders/CvrpDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/CvrpDecoder.cu
  ${CMAKE_CURRENT_LIST_DIR}/decoders/CvrpDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/ScpDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/ScpDecoder.cu
  ${CMAKE_CURRENT_LIST_DIR}/decoders/ScpDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/TspDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/TspDecoder.cu
  ${CMAKE_CURRENT_LIST_DIR}/decoders/TspDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/main.cu
)

set_target_properties(gpu-brkga PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(gpu-brkga PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET gpu-brkga PROPERTY CUDA_STANDARD 11)
set_property(TARGET gpu-brkga PROPERTY CUDA_ARCHITECTURES 61)

target_compile_options(gpu-brkga PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp>)
target_link_libraries(gpu-brkga PUBLIC OpenMP::OpenMP_CXX PUBLIC -lcurand -fopenmp)

# The executable of the fixed version
add_executable(gpu-brkga-fix
  ${CUDA_COMMON_SRC}
  ${CMAKE_CURRENT_LIST_DIR}/decoders/CvrpDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/CvrpDecoder.cu
  ${CMAKE_CURRENT_LIST_DIR}/decoders/CvrpDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/ScpDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/ScpDecoder.cu
  ${CMAKE_CURRENT_LIST_DIR}/decoders/ScpDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/TspDecoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/TspDecoder.cu
  ${CMAKE_CURRENT_LIST_DIR}/decoders/TspDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/main.cu
)

set_target_properties(gpu-brkga-fix PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(gpu-brkga-fix PROPERTIES COMPILE_DEFINITIONS "GPU_BRKGA_FIXED=1")
set_target_properties(gpu-brkga-fix PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET gpu-brkga-fix PROPERTY CUDA_STANDARD 11)
set_property(TARGET gpu-brkga-fix PROPERTY CUDA_ARCHITECTURES 61)

target_compile_options(gpu-brkga-fix PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp>)
target_link_libraries(gpu-brkga-fix PUBLIC OpenMP::OpenMP_CXX PUBLIC -lcurand -fopenmp)

# gpu-brkga
include_directories(SYSTEM ${GPU_BRKGA_ROOT})

add_library(gpu-brkga-lib INTERFACE
  ${GPU_BRKGA_ROOT}/GPU-BRKGA/cuda_errorchecking.h
  ${GPU_BRKGA_ROOT}/GPU-BRKGA/GPUBRKGA.cuh
  ${GPU_BRKGA_ROOT}/GPU-BRKGA/Individual.h
  ${GPU_BRKGA_ROOT}/GPU-BRKGA/kernels.cuh
)

set_property(TARGET gpu-brkga PROPERTY CUDA_STANDARD 11)
set_property(TARGET gpu-brkga PROPERTY CUDA_ARCHITECTURES 61)
set_target_properties(gpu-brkga PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(gpu-brkga PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(gpu-brkga LINK_PUBLIC gpu-brkga-lib)
