cmake_minimum_required(VERSION 3.10)
project(yelmewad2021 LANGUAGES CUDA C)
set(CMAKE_CXX_STANDARD 17)

add_executable(yelmewad2021-sequential src/sequential_cvrp.c)
target_link_libraries(yelmewad2021-sequential m)

add_executable(yelmewad2021-customer src/customer_level.cu)
set_property(TARGET yelmewad2021-customer PROPERTY CUDA_STANDARD 17)
set_property(TARGET yelmewad2021-customer PROPERTY CUDA_ARCHITECTURES 61)

add_executable(yelmewad2021-route src/route_level.cu)
set_property(TARGET yelmewad2021-route PROPERTY CUDA_STANDARD 17)
set_property(TARGET yelmewad2021-route PROPERTY CUDA_ARCHITECTURES 61)
