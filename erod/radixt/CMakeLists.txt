## http://www.cmake.org/cmake/help/v2.8.12/cmake.html#module:FindCUDA

INCLUDE (${APHID_DIR}/cmake/cuda_config.cmake)

include_directories (${APHID_DIR}/radixsort
${APHID_DIR}/reduction
${APHID_DIR}/cudabvh
${APHID_DIR}/scan)

set (HEADER_FILES
${APHID_DIR}/reduction/CudaReduction.h)
set (SOURC_FILES main.cpp)
set (CUSOURCES ${APHID_DIR}/radixsort/radixsort.cu)

CUDA_ADD_EXECUTABLE (radixst ${HEADER_FILES} ${SOURC_FILES} ${CUSOURCES})

target_link_libraries(radixst ${CUDA_CUT_LIBRARY})
target_link_libraries(radixst ${CUDA_CUDART_LIBRARY})

## boost must after nvcc for some reason
INCLUDE (${APHID_DIR}/cmake/boost_config.cmake)

target_link_libraries (radixst ${Boost_SYSTEM_LIBRARY})
target_link_libraries (radixst ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (radixst ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (radixst ${Boost_REGEX_LIBRARY})

target_link_libraries (radixst ${APHID_FOUND_LIBRARY})
target_link_libraries (radixst ${APHID_MATH_LIBRARY})

