cmake_minimum_required(VERSION 3.16)
project(BRKGA)

set(CMAKE_CXX_STANDARD 17)


add_compile_definitions(LOG_LEVEL=1)
add_compile_definitions(eps=1e-6)


# ===== Dependencies =====
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIR})
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
set(CMAKE_CXX_CLANG_TIDY clang-tidy)


# ===== The targets =====
add_library(brkga
        src/brkga/Brkga.cpp
        src/brkga/Brkga.hpp
        src/brkga/BrkgaOpenCL.hpp
        src/brkga/Problem.hpp
        src/brkga/IO.hpp
        src/brkga/IO.cpp
        src/brkga/Configuration.cpp
        src/brkga/Configuration.hpp
        src/brkga/OpenCL.cpp
        src/brkga/OpenCL.hpp
)
target_link_libraries(brkga ${OpenCL_LIBRARY})

add_executable(tsp
        src/tsp/main.cpp
        src/tsp/Point.hpp
        src/tsp/TspProblem.cpp
        src/tsp/TspProblem.hpp
)
target_link_libraries(tsp LINK_PUBLIC brkga)
