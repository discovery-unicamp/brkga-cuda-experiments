cmake_minimum_required(VERSION 3.10)

project(brkga_cuda)

find_package(OpenMP)
find_package(CUDA REQUIRED)
enable_language(CUDA)
include_directories("${CUDA_INCLUDE_DIRS}")
 
file(GLOB
  CHECK_SOURCE_FILES
  brkga_cuda_api/*.cpp brkga_cuda_api/*.cu brkga_cuda_api/*.h brkga_cuda_api/*.cuh
  tsp_example/src/*.cpp tsp_example/src/*.h
)
include("cmake/ClangFormat.cmake")

set(API_FILES
  ../brkga_cuda_api/BRKGA.cu
  ../brkga_cuda_api/ConfigFile.cpp
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)

add_subdirectory(tsp_example)
