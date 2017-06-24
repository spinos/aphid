## http://www.cmake.org/cmake/help/v2.8.12/cmake.html#module:FindCUDA

INCLUDE (${APHID_DIR}/cmake/cuda_config.cmake)

include_directories (${APHID_DIR}/radixsort
${APHID_DIR}/reduction
${APHID_DIR}/cudabvh
${APHID_DIR}/scan)

set (HEADER_FILES ${APHID_DIR}/shared/Morton3D.h
${APHID_DIR}/reduction/CudaReduction.h)
set (SOURC_FILES main.cpp
${APHID_DIR}/shared/CudaBase.cpp
${APHID_DIR}/shared/CUDABuffer.cpp
${APHID_DIR}/scan/CudaScan.cpp
${APHID_DIR}/reduction/CudaReduction.cpp)
set (CUSOURCES ${APHID_DIR}/radixsort/radixsort.cu
 cu_testAtomic.cu
${APHID_DIR}/scan/scan.cu
${APHID_DIR}/reduction/cuReduceMax.cu
${APHID_DIR}/reduction/cuReduceMin.cu
${APHID_DIR}/reduction/cuReduceMinMax.cu
${APHID_DIR}/reduction/cuReduceMinMaxBox.cu
${APHID_DIR}/reduction/cuReduceSum.cu
TestPRNG.cu)

CUDA_ADD_EXECUTABLE (testcu ${HEADER_FILES} ${SOURC_FILES} ${CUSOURCES})

## boost must after nvcc for some reason
INCLUDE (${APHID_DIR}/cmake/boost_config.cmake)

target_link_libraries(testcu ${CUDA_CUDART_LIBRARY})
target_link_libraries(testcu ${CUDA_cublas_LIBRARY})
target_link_libraries(testcu ${CUDA_cusparse_LIBRARY})

target_link_libraries (testcu ${Boost_SYSTEM_LIBRARY})
target_link_libraries (testcu ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (testcu ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (testcu ${Boost_REGEX_LIBRARY})

target_link_libraries (testcu ${APHID_FOUND_LIBRARY})
target_link_libraries (testcu ${APHID_MATH_LIBRARY})

