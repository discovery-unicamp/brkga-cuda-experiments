cmake_minimum_required(VERSION 3.10)
project(brkga-tsp LANGUAGES CUDA CXX)

add_executable(brkga-tsp
  src/main.cpp
  src/TSPInstance.cpp
  src/TSPInstance.cu
  src/TSPInstance.h
)

find_package(OpenMP REQUIRED)
target_link_libraries(brkga-tsp PRIVATE OpenMP::OpenMP_CXX)

set_target_properties(brkga-tsp PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(brkga-tsp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET brkga-tsp PROPERTY CUDA_STANDARD 17)
set_property(TARGET brkga-tsp PROPERTY CUDA_ARCHITECTURES 61)

target_link_libraries(brkga-tsp LINK_PUBLIC brkga-cuda)
target_link_libraries(brkga-tsp LINK_PUBLIC gpu-brkga)

if (${BRKGA_OPENCL_ENABLED})
  target_link_libraries(brkga-tsp LINK_PUBLIC brkga-opencl)
endif()
