cmake_minimum_required(VERSION 3.16)
project(brkga-optimizer LANGUAGES CUDA CXX)

# Configuration
set(CMAKE_CXX_STANDARD 11)
set(BRKGA_OPENCL_ENABLED false)

add_compile_options(
  -Wall -Wextra -Wshadow -Wformat=2 -Wfloat-equal -Wshift-overflow
  -Wconversion -Wno-sign-conversion
)

# Dependencies
find_package(OpenMP REQUIRED)

# brkga-cuda
set(BRKGA_CUDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../brkga-cuda-itor")
include("${BRKGA_CUDA_ROOT}/CMakeLists.txt")
include_directories(${BRKGA_CUDA_ROOT})

# gpu-brkga
set(GPU_BRKGA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../GPU-BRKGA")
include("${GPU_BRKGA_ROOT}/CMakeLists.txt")
include_directories(SYSTEM ${GPU_BRKGA_ROOT})

# brkga-opencl
if (${BRKGA_OPENCL_ENABLED})
  message("Enabling BRKGA OpenCL")
  set(BRKGA_OPENCL_ROOT "${CMAKE_CURRENT_LIST_DIR}/../brkga-opencl")
  include("${BRKGA_OPENCL_ROOT}/CMakeLists.txt")
  add_compile_definitions(BRKGA_OPENCL_ENABLED)
  include_directories(${BRKGA_OPENCL_ROOT})
endif()

add_executable(brkga-optimizer
  src/instances/CvrpInstance.cpp
  src/instances/CvrpInstance.cu
  src/instances/CvrpInstance.hpp
  src/instances/TSPInstance.cpp
  src/instances/TSPInstance.cu
  src/instances/TSPInstance.hpp
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

if (${BRKGA_OPENCL_ENABLED})
  target_link_libraries(brkga-optimizer LINK_PUBLIC brkga-opencl)
endif()
