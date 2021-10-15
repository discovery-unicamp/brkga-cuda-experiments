include_directories(../brkga_cuda_api)
set(CMAKE_CXX_STANDARD 17)
include_directories(src)

add_executable(brkga-cvrp
        src/main.cpp
        src/CvrpInstance.cu
        src/CvrpInstance.hpp
        src/Point.hpp
        ${API_FILES}
)

set_target_properties(brkga-cvrp PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(brkga-cvrp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET brkga-cvrp PROPERTY CUDA_STANDARD 17)
set_property(TARGET brkga-cvrp PROPERTY CUDA_ARCHITECTURES 61)
target_compile_options(brkga-cvrp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler
        -fopenmp
>)

if(OpenMP_CXX_FOUND)
    if(USE_NVTX)
        target_link_libraries(brkga-cvrp PUBLIC OpenMP::OpenMP_CXX PUBLIC ${LIBNVTOOLSEXT} -lcurand -fopenmp)
    else()
        target_link_libraries(brkga-cvrp PUBLIC OpenMP::OpenMP_CXX PUBLIC -lcurand -fopenmp)
    endif()
else()
    target_link_libraries(brkga-cvrp PUBLIC -lcurand)
endif()
