cmake_minimum_required(VERSION 3.16)
project(BRKGA)

set(CMAKE_CXX_STANDARD 11)


add_compile_definitions(LOG_LEVEL=1)
add_compile_definitions(eps=1e-6)


# ===== Dependencies =====
find_package(OpenCL REQUIRED)
include_directories(SYSTEM ${OpenCL_INCLUDE_DIR})
add_compile_definitions(CL_HPP_ENABLE_EXCEPTIONS)
add_compile_definitions(CL_HPP_TARGET_OPENCL_VERSION=210)


# ===== Code optimization =====
# -Ofast enables non standard optimizations.
# -march compiles to the current CPU architecture, which may not work on another architecture.
# -flto enables link time optimization.
# -frename-registers avoids false dependencies by using registers.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -flto -frename-registers")


# ===== Runtime check =====
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -D_FORTIFY_SOURCE=2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-sanitize-recover=all -fstack-protector")


# ===== Code warnings =====
add_compile_options(-Wall -Wextra -Wshadow -Wformat=2 -Wfloat-equal -Wconversion -Wno-sign-conversion -Wshift-overflow -Wcast-qual -Wcast-align)
# set(CMAKE_CXX_CLANG_TIDY clang-tidy)


# ===== The targets =====
add_library(brkga-opencl
        ${CMAKE_CURRENT_LIST_DIR}/brkga-opencl/Brkga.cpp
        ${CMAKE_CURRENT_LIST_DIR}/brkga-opencl/Brkga.hpp
        ${CMAKE_CURRENT_LIST_DIR}/brkga-opencl/BrkgaOpenCL.hpp
        ${CMAKE_CURRENT_LIST_DIR}/brkga-opencl/Configuration.cpp
        ${CMAKE_CURRENT_LIST_DIR}/brkga-opencl/Configuration.hpp
        ${CMAKE_CURRENT_LIST_DIR}/brkga-opencl/IO.hpp
        ${CMAKE_CURRENT_LIST_DIR}/brkga-opencl/IO.cpp
        ${CMAKE_CURRENT_LIST_DIR}/brkga-opencl/OpenCL.cpp
        ${CMAKE_CURRENT_LIST_DIR}/brkga-opencl/OpenCL.hpp
        ${CMAKE_CURRENT_LIST_DIR}/brkga-opencl/Problem.hpp
)

target_link_libraries(brkga-opencl ${OpenCL_LIBRARY})
