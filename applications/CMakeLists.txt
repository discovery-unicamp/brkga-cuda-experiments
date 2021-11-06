cmake_minimum_required(VERSION 3.10)
project(brkga_applications LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall -Wextra -Wshadow -Wformat=2 -Wfloat-equal -Wconversion -Wno-sign-conversion -Wshift-overflow -Wcast-qual -Wcast-align)

set(BRKGA_CUDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../brkga-cuda-itor")
set(GPU_BRKGA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../GPU-BRKGA/")

# brkga-cuda
include("${BRKGA_CUDA_ROOT}/CMakeLists.txt")
include_directories(${BRKGA_CUDA_ROOT})

# gpu-brkga
include("${GPU_BRKGA_ROOT}/CMakeLists.txt")
include_directories(SYSTEM ${GPU_BRKGA_ROOT})

add_subdirectory(cvrp)
