cmake_minimum_required(VERSION 3.10)
project(brkga-cvrp)
set(CMAKE_CXX_STANDARD 17)

include("${CMAKE_CURRENT_LIST_DIR}/../../brkga-cuda-itor/CMakeLists.txt")
include_directories("${CMAKE_CURRENT_LIST_DIR}/../../brkga-cuda-itor")

add_compile_options(-Wall -Wextra -Wshadow -Wformat=2 -Wfloat-equal -Wconversion -Wno-sign-conversion -Wshift-overflow -Wcast-qual -Wcast-align)

add_executable(brkga-cvrp
        src/main.cpp
        src/CvrpInstance.cu
        src/CvrpInstance.hpp
        src/Point.hpp
)

set_target_properties(brkga-cvrp PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(brkga-cvrp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET brkga-cvrp PROPERTY CUDA_STANDARD 17)
set_property(TARGET brkga-cvrp PROPERTY CUDA_ARCHITECTURES 61)
target_compile_options(brkga-cvrp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler
        -fopenmp
>)

target_link_libraries(brkga-cvrp LINK_PUBLIC brkga-cuda)

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
