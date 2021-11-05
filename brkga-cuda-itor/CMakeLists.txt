cmake_minimum_required(VERSION 3.10)

project(brkga_cuda LANGUAGES CUDA CXX)

find_package(OpenMP)
find_package(CUDA REQUIRED)
enable_language(CUDA)
include_directories("${CUDA_INCLUDE_DIRS}")

if(USE_NVTX)
 message("Compiling to use NVTX!")
 add_definitions(-DUSE_NVTX)
 find_library(LIBNVTOOLSEXT nvToolsExt PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

add_library(brkga-cuda
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/Instance.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/BRKGA.cu
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/ConfigFile.cpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/MathUtils.h
)

set_target_properties(brkga-cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(brkga-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET brkga-cuda PROPERTY CUDA_STANDARD 17)
set_property(TARGET brkga-cuda PROPERTY CUDA_ARCHITECTURES 61)
target_compile_options(brkga-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp>)

set_target_properties(brkga-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
