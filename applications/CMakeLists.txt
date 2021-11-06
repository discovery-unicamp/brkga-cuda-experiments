cmake_minimum_required(VERSION 3.10)
project(brkga_applications LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall -Wextra -Wshadow -Wformat=2 -Wfloat-equal -Wconversion -Wno-sign-conversion -Wshift-overflow -Wcast-qual -Wcast-align)

set(BRKGA_OPENCL_ENABLED false)

# brkga-cuda
set(BRKGA_CUDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../brkga-cuda-itor")
include("${BRKGA_CUDA_ROOT}/CMakeLists.txt")
include_directories(${BRKGA_CUDA_ROOT})

# brkga-opencl
set(GPU_BRKGA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../GPU-BRKGA/")
include("${GPU_BRKGA_ROOT}/CMakeLists.txt")
include_directories(SYSTEM ${GPU_BRKGA_ROOT})

# gpu-brkga
if (${BRKGA_OPENCL_ENABLED})
        message("Enabling BRKGA OpenCL")
        set(BRKGA_OPENCL_ROOT "${CMAKE_CURRENT_LIST_DIR}/../brkga-opencl")
        include("${BRKGA_OPENCL_ROOT}/CMakeLists.txt")
        add_compile_definitions(BRKGA_OPENCL_ENABLED)
        include_directories(${BRKGA_OPENCL_ROOT})
endif()

add_subdirectory(cvrp)
