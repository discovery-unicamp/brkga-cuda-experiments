cmake_minimum_required(VERSION 3.10)
project(brkga-cvrp LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)

add_compile_definitions(RUN_BRKGA_CUDA)
add_compile_options(-Wall -Wextra -Wshadow -Wformat=2 -Wfloat-equal -Wconversion -Wno-sign-conversion -Wshift-overflow -Wcast-qual -Wcast-align)

set(BRKGA_CUDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../brkga-cuda-itor")
set(GPU_BRKGA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../GPU-BRKGA/")

add_executable(brkga-cvrp
        src/main.cpp
        src/CvrpInstance.cu
        src/CvrpInstance.hpp
        src/Point.hpp
        src/brkga/BaseBrkga.hpp
        src/brkga/BrkgaCuda.cpp
        src/brkga/BrkgaCuda.hpp
        src/brkga/GpuBrkga.cpp
        src/brkga/GpuBrkga.hpp
)

set_source_files_properties(src/brkga/GpuBrkga.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/brkga/GpuBrkga.hpp PROPERTIES LANGUAGE CUDA)

set_target_properties(brkga-cvrp PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(brkga-cvrp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET brkga-cvrp PROPERTY CUDA_STANDARD 17)
set_property(TARGET brkga-cvrp PROPERTY CUDA_ARCHITECTURES 61)
target_compile_options(brkga-cvrp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp>)

# brkga-cuda
include("${BRKGA_CUDA_ROOT}/CMakeLists.txt")
include_directories(${BRKGA_CUDA_ROOT})
target_link_libraries(brkga-cvrp LINK_PUBLIC brkga-cuda)

# gpu-brkga
include("${GPU_BRKGA_ROOT}/CMakeLists.txt")
include_directories(SYSTEM ${GPU_BRKGA_ROOT})
target_link_libraries(brkga-cvrp LINK_PUBLIC gpu-brkga)

# FIXME não funciona sem open mp
if(OpenMP_CXX_FOUND)
  if(USE_NVTX)
    target_link_libraries(brkga-cvrp PUBLIC OpenMP::OpenMP_CXX PUBLIC ${LIBNVTOOLSEXT} -lcurand -fopenmp)
  else()
    target_link_libraries(brkga-cvrp PUBLIC OpenMP::OpenMP_CXX PUBLIC -lcurand -fopenmp)
  endif()
else()
  target_link_libraries(brkga-cvrp PUBLIC -lcurand)
endif()
