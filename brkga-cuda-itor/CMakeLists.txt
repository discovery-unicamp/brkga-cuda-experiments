cmake_minimum_required(VERSION 3.10)
project(brkga_cuda LANGUAGES CUDA CXX)

set(BB_SEGSORT_ROOT "${CMAKE_CURRENT_LIST_DIR}/bb_segsort")
include("${BB_SEGSORT_ROOT}/CMakeLists.txt")
include_directories(SYSTEM "${BB_SEGSORT_ROOT}")

find_package(OpenMP)
find_package(CUDA REQUIRED)
enable_language(CUDA)
include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}")

add_compile_definitions(LOG_LEVEL=2)
# add_compile_definitions(BRKGA_DEBUG)

if(USE_NVTX)
  message("Compiling to use NVTX!")
  add_definitions(-DUSE_NVTX)
  find_library(LIBNVTOOLSEXT nvToolsExt PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

add_library(brkga-cuda
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/BBSegSort.cuh
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/BRKGA.cu
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/BRKGA.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/BrkgaConfiguration.cpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/BrkgaConfiguration.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/CudaContainers.cuh
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/CudaError.cuh
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/DecodeType.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/Logger.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/Instance.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga_cuda_api/MathUtils.hpp
)

set_target_properties(brkga-cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(brkga-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET brkga-cuda PROPERTY CUDA_STANDARD 17)
set_property(TARGET brkga-cuda PROPERTY CUDA_ARCHITECTURES 61)

target_compile_options(brkga-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp>)
target_link_libraries(brkga-cuda PUBLIC bb_segsort)

# FIXME doesn't work without OpenMP
if(OpenMP_CXX_FOUND)
  if(USE_NVTX)
    target_link_libraries(brkga-cuda PUBLIC OpenMP::OpenMP_CXX PUBLIC ${LIBNVTOOLSEXT} -lcurand -fopenmp)
  else()
    target_link_libraries(brkga-cuda PUBLIC OpenMP::OpenMP_CXX PUBLIC -lcurand -fopenmp)
  endif()
else()
  target_link_libraries(brkga-cuda PUBLIC -lcurand)
endif()
