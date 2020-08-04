include_directories(../brkga_cuda_api)
include_directories(src)

add_executable(brkga-tsp
  src/main.cpp
  src/TSPInstance.cpp
  src/Decoder.cu
  ${API_FILES}
)

set_target_properties(brkga-tsp PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(brkga-tsp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(brkga-tsp ${CUDA_LIBRARIES} m stdc++)
 
target_link_libraries(brkga-tsp -fopenmp -lcurand)
