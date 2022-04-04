cmake_minimum_required(VERSION 3.10)
project(brkga-cvrp LANGUAGES CUDA CXX)

add_executable(brkga-cvrp
  src/instances/CvrpInstance.cpp
  src/instances/CvrpInstance.cu
  src/instances/CvrpInstance.hpp
  src/instances/TSPInstance.cpp
  src/instances/TSPInstance.cu
  src/instances/TSPInstance.hpp
  src/Checker.hpp
  src/GpuBrkgaWrapper.cu
  src/GpuBrkgaWrapper.hpp
  src/main.cpp
  src/MinQueue.hpp
  src/Point.cpp
  src/Point.hpp
)

find_package(OpenMP REQUIRED)
target_link_libraries(brkga-cvrp PRIVATE OpenMP::OpenMP_CXX)

set_target_properties(brkga-cvrp PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(brkga-cvrp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET brkga-cvrp PROPERTY CUDA_STANDARD 17)
set_property(TARGET brkga-cvrp PROPERTY CUDA_ARCHITECTURES 61)

target_link_libraries(brkga-cvrp LINK_PUBLIC brkga-cuda)
target_link_libraries(brkga-cvrp LINK_PUBLIC gpu-brkga)

if (${BRKGA_OPENCL_ENABLED})
  target_link_libraries(brkga-cvrp LINK_PUBLIC brkga-opencl)
endif()
