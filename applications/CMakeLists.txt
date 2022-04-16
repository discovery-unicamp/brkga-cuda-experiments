cmake_minimum_required(VERSION 3.16)
project(brkga-optimizer LANGUAGES CUDA CXX)

# Configuration
set(CMAKE_CXX_STANDARD 11)
set(BRKGA_OPENCL_ENABLED false)

add_compile_options(
  -Wall -Wextra -Wshadow -Wformat=2 -Wfloat-equal -Wshift-overflow
  -Wconversion -Wno-sign-conversion
)
add_compile_definitions(LOG_LEVEL=3)

# Dependencies
find_package(OpenMP REQUIRED)

find_package(CUDA REQUIRED)
include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}")

# brkga-cuda
set(BRKGA_CUDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../brkga-cuda-itor")
include("${BRKGA_CUDA_ROOT}/CMakeLists.txt")
include_directories(${BRKGA_CUDA_ROOT})

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

# brkga-opencl
if (${BRKGA_OPENCL_ENABLED})
  message("Enabling BRKGA OpenCL")
  set(BRKGA_OPENCL_ROOT "${CMAKE_CURRENT_LIST_DIR}/../brkga-opencl")
  include("${BRKGA_OPENCL_ROOT}/CMakeLists.txt")
  add_compile_definitions(BRKGA_OPENCL_ENABLED)
  include_directories(${BRKGA_OPENCL_ROOT})
endif()

# brkga-api
set(BRKGA_API_ROOT "${CMAKE_CURRENT_LIST_DIR}/../toso-brkga-api")
include_directories(SYSTEM ${BRKGA_API_ROOT})

add_library(brkga-api INTERFACE
  ${BRKGA_API_ROOT}/brkgaAPI/BRKGA.h
  ${BRKGA_API_ROOT}/brkgaAPI/MTRand.h
  ${BRKGA_API_ROOT}/brkgaAPI/Population.cpp
  ${BRKGA_API_ROOT}/brkgaAPI/Population.h
)

add_executable(brkga-optimizer
  src/instances/CvrpInstance.cpp
  src/instances/CvrpInstance.cu
  src/instances/CvrpInstance.hpp
  src/instances/TSPInstance.cpp
  src/instances/TSPInstance.cu
  src/instances/TSPInstance.hpp
  src/wrapper/BaseWrapper.cu
  src/wrapper/BaseWrapper.hpp
  src/wrapper/BrkgaApiWrapper.cpp
  src/wrapper/BrkgaApiWrapper.hpp
  src/wrapper/BrkgaCudaWrapper.cu
  src/wrapper/BrkgaCudaWrapper.hpp
  src/wrapper/GpuBrkgaWrapper.cu
  src/wrapper/GpuBrkgaWrapper.hpp
  src/Checker.hpp
  src/main.cpp
  src/MinQueue.hpp
  src/Point.cpp
  src/Point.hpp
)

target_link_libraries(brkga-optimizer PRIVATE OpenMP::OpenMP_CXX)

set_target_properties(brkga-optimizer PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(brkga-optimizer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET brkga-optimizer PROPERTY CUDA_STANDARD 11)
set_property(TARGET brkga-optimizer PROPERTY CUDA_ARCHITECTURES 61)

target_link_libraries(brkga-optimizer LINK_PUBLIC brkga-cuda)
target_link_libraries(brkga-optimizer LINK_PUBLIC gpu-brkga)
target_link_libraries(brkga-optimizer LINK_PUBLIC brkga-api)

if (${BRKGA_OPENCL_ENABLED})
  target_link_libraries(brkga-optimizer LINK_PUBLIC brkga-opencl)
endif()
