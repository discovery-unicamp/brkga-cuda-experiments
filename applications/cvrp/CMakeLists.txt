cmake_minimum_required(VERSION 3.10)
project(brkga-cvrp LANGUAGES CUDA CXX)

add_executable(brkga-cvrp
  src/main.cpp
  src/CvrpInstance.cu
  src/CvrpInstance.hpp
  src/Point.hpp
  src/brkga/BaseBrkga.hpp
  src/brkga/BrkgaCuda.cpp
  src/brkga/BrkgaCuda.hpp
  src/brkga/BrkgaOpenCL.cpp
  src/brkga/BrkgaOpenCL.hpp
  src/brkga/GpuBrkga.cpp
  src/brkga/GpuBrkga.hpp
)

set_source_files_properties(src/brkga/GpuBrkga.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/brkga/GpuBrkga.hpp PROPERTIES LANGUAGE CUDA)

set_target_properties(brkga-cvrp PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(brkga-cvrp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET brkga-cvrp PROPERTY CUDA_STANDARD 17)
set_property(TARGET brkga-cvrp PROPERTY CUDA_ARCHITECTURES 61)

target_link_libraries(brkga-cvrp LINK_PUBLIC brkga-cuda)
target_link_libraries(brkga-cvrp LINK_PUBLIC gpu-brkga)

if (${BRKGA_OPENCL_ENABLED})
        target_link_libraries(brkga-cvrp LINK_PUBLIC brkga-opencl)
endif()
