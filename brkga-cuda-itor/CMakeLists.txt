cmake_minimum_required(VERSION 3.10)

project(brkga_cuda)

find_package(OpenMP)
find_package(CUDA REQUIRED)
enable_language(CUDA)
include_directories("${CUDA_INCLUDE_DIRS}")

if(USE_NVTX)
 message("Compiling to use NVTX!")
 add_definitions(-DUSE_NVTX)
 find_library(LIBNVTOOLSEXT nvToolsExt PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
 
file(GLOB
  CHECK_SOURCE_FILES
  brkga_cuda_api/*.cpp brkga_cuda_api/*.cu brkga_cuda_api/*.h brkga_cuda_api/*.cuh
  tsp_example/src/*.cpp tsp_example/src/*.h
)
include("cmake/ClangFormat.cmake")

set(API_FILES
  ../brkga_cuda_api/Instance.hpp
  ../brkga_cuda_api/BRKGA.cu
  ../brkga_cuda_api/ConfigFile.cpp
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -Wformat=2 -Wfloat-equal -Wconversion -Wno-sign-conversion -Wshift-overflow -Wcast-qual -Wcast-align")

add_subdirectory(tsp_example)
add_subdirectory(scp_example)
add_subdirectory(cvrp_example)
