cmake_minimum_required(VERSION 3.16)
PROJECT(brkga-cuda-1.0-optimizer LANGUAGES CUDA CXX)

# Dependencies
find_package(OpenMP REQUIRED)

find_package(CUDA REQUIRED)
enable_language(CUDA)
include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}")

# The executable
add_executable(brkga-cuda-1.0
  ${CUDA_COMMON_SRC}
  ${CMAKE_CURRENT_LIST_DIR}/decoders/CvrpDecoder.cu
  ${CMAKE_CURRENT_LIST_DIR}/decoders/CvrpDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/ScpDecoder.cu
  ${CMAKE_CURRENT_LIST_DIR}/decoders/ScpDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/TspDecoder.cu
  ${CMAKE_CURRENT_LIST_DIR}/decoders/TspDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/BrkgaCuda.cu
  ${CMAKE_CURRENT_LIST_DIR}/BrkgaCuda.hpp
  ${CMAKE_CURRENT_LIST_DIR}/BrkgaCudaInclude.cu
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
)

set_target_properties(brkga-cuda-1.0 PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(brkga-cuda-1.0 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET brkga-cuda-1.0 PROPERTY CUDA_STANDARD 11)
set_property(TARGET brkga-cuda-1.0 PROPERTY CUDA_ARCHITECTURES 61)

target_compile_options(brkga-cuda-1.0 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp>)
target_link_libraries(brkga-cuda-1.0 PUBLIC OpenMP::OpenMP_CXX PUBLIC -lcurand -fopenmp)

# The library
include_directories(SYSTEM ${BRKGA_CUDA_1_ROOT})
