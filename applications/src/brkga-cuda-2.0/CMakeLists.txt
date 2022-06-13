cmake_minimum_required(VERSION 3.16)
PROJECT(brkga-cuda-2.0-optimizer LANGUAGES CUDA CXX)

# Dependencies
find_package(OpenMP REQUIRED)

find_package(CUDA REQUIRED)
enable_language(CUDA)
include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}")

# The executable
add_executable(brkga-cuda-2.0
  ${CUDA_COMMON_SRC}
  ${CMAKE_CURRENT_LIST_DIR}/decoders/CvrpDecoder.cu
  ${CMAKE_CURRENT_LIST_DIR}/decoders/CvrpDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/ScpDecoder.cu
  ${CMAKE_CURRENT_LIST_DIR}/decoders/ScpDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/decoders/TspDecoder.cu
  ${CMAKE_CURRENT_LIST_DIR}/decoders/TspDecoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
)

set_target_properties(brkga-cuda-2.0 PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(brkga-cuda-2.0 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET brkga-cuda-2.0 PROPERTY CUDA_STANDARD 11)
set_property(TARGET brkga-cuda-2.0 PROPERTY CUDA_ARCHITECTURES 61)

target_compile_options(brkga-cuda-2.0 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp>)
target_link_libraries(brkga-cuda-2.0 PUBLIC OpenMP::OpenMP_CXX PUBLIC -lcurand -fopenmp)

# brkga-cuda
include_directories(${BRKGA_CUDA_ROOT})
include("${BRKGA_CUDA_ROOT}/CMakeLists.txt")
target_link_libraries(brkga-cuda-2.0 LINK_PUBLIC brkga-cuda)
