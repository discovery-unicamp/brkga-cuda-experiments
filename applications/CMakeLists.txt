cmake_minimum_required(VERSION 3.16)
project(brkga-optimizer LANGUAGES CUDA CXX)

# Configuration
set(CMAKE_CXX_STANDARD 11)
set(BRKGA_OPENCL_ENABLED false)

add_compile_options(
  -Wall -Wextra -Wshadow -Wformat=2 -Wformat-truncation -Wfloat-equal
  -Wshift-overflow -Wconversion -Wno-sign-conversion -Wundef -fno-common
)
add_compile_definitions(LOG_LEVEL=3)

# Dependencies
find_package(OpenMP REQUIRED)

find_package(CUDA REQUIRED)
enable_language(CUDA)
include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}")

# The executable
add_executable(brkga-optimizer
  src/instances/CvrpInstance.cpp
  src/instances/CvrpInstance.cu
  src/instances/CvrpInstance.hpp
  src/instances/Instance.cpp
  src/instances/Instance.hpp
  src/instances/ScpInstance.cpp
  src/instances/ScpInstance.cu
  src/instances/ScpInstance.hpp
  src/instances/TspInstance.cpp
  src/instances/TspInstance.cu
  src/instances/TspInstance.hpp
  src/utils/StringUtils.cpp
  src/utils/StringUtils.hpp
  src/wrapper/BaseWrapper.cu
  src/wrapper/BaseWrapper.hpp
  src/wrapper/BrkgaApiWrapper.cpp
  src/wrapper/BrkgaApiWrapper.hpp
  src/wrapper/BrkgaCudaWrapper.cu
  src/wrapper/BrkgaCudaWrapper.hpp
  src/wrapper/GpuBrkgaWrapper.cu
  src/wrapper/GpuBrkgaWrapper.hpp
  src/wrapper/OldBrkgaCudaWrapper.cu
  src/wrapper/OldBrkgaCudaWrapper.hpp
  src/Checker.hpp
  src/main.cpp
  src/MinQueue.hpp
  src/Point.cpp
  src/Point.hpp
)

set_target_properties(brkga-optimizer PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(brkga-optimizer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET brkga-optimizer PROPERTY CUDA_STANDARD 11)
set_property(TARGET brkga-optimizer PROPERTY CUDA_ARCHITECTURES 61)

target_compile_options(brkga-optimizer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp>)
target_link_libraries(brkga-optimizer PUBLIC OpenMP::OpenMP_CXX PUBLIC -lcurand -fopenmp)

# brkga-cuda
set(BRKGA_CUDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../brkga-cuda-itor")
include_directories(${BRKGA_CUDA_ROOT})

include("${BRKGA_CUDA_ROOT}/CMakeLists.txt")
target_link_libraries(brkga-optimizer LINK_PUBLIC brkga-cuda)

# old version of the brkga-cuda
set(OLD_BRKGA_CUDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../brkga-cuda-wscad")
include_directories(SYSTEM ${OLD_BRKGA_CUDA_ROOT})

# gpu-brkga
set(GPU_BRKGA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../GPU-BRKGA")
include_directories(SYSTEM ${GPU_BRKGA_ROOT})

add_library(gpu-brkga INTERFACE
  ${GPU_BRKGA_ROOT}/GPU-BRKGA/cuda_errorchecking.h
  ${GPU_BRKGA_ROOT}/GPU-BRKGA/GPUBRKGA.cuh
  ${GPU_BRKGA_ROOT}/GPU-BRKGA/Individual.h
  ${GPU_BRKGA_ROOT}/GPU-BRKGA/kernels.cuh
)

set_property(TARGET gpu-brkga PROPERTY CUDA_STANDARD 11)
set_property(TARGET gpu-brkga PROPERTY CUDA_ARCHITECTURES 61)
set_target_properties(gpu-brkga PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(gpu-brkga PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(brkga-optimizer LINK_PUBLIC gpu-brkga)

# brkga-api
set(BRKGA_API_ROOT "${CMAKE_CURRENT_LIST_DIR}/../toso-brkga-api")
include_directories(SYSTEM ${BRKGA_API_ROOT})

add_library(brkga-api INTERFACE
  ${BRKGA_API_ROOT}/brkgaAPI/BRKGA.h
  ${BRKGA_API_ROOT}/brkgaAPI/MTRand.h
  ${BRKGA_API_ROOT}/brkgaAPI/Population.cpp
  ${BRKGA_API_ROOT}/brkgaAPI/Population.h
)

target_link_libraries(brkga-optimizer LINK_PUBLIC brkga-api)

# brkga-opencl
if (${BRKGA_OPENCL_ENABLED})
  message("Enabling BRKGA OpenCL")
  add_compile_definitions(BRKGA_OPENCL_ENABLED)
  set(BRKGA_OPENCL_ROOT "${CMAKE_CURRENT_LIST_DIR}/../brkga-opencl")
  include_directories(${BRKGA_OPENCL_ROOT})
  include("${BRKGA_OPENCL_ROOT}/CMakeLists.txt")
  target_link_libraries(brkga-optimizer LINK_PUBLIC brkga-opencl)
endif()
