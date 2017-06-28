INCLUDE (${APHID_DIR}/cmake/cuda_config.cmake)

include_directories (${APHID_DIR}/radixsort
${APHID_DIR}/reduction
${APHID_DIR}/cudabvh
${APHID_DIR}/scan
${APHID_DIR}/shared
${APHID_DIR}/sahbvh
${APHID_DIR}/onebitsort
${APHID_DIR}/taskQueue)

set (HEADER_FILES ${APHID_DIR}/shared/Morton3D.h
${APHID_DIR}/reduction/CudaReduction.h)
set (SOURC_FILES main.cpp
${APHID_DIR}/shared/CudaBase.cpp
${APHID_DIR}/shared/CUDABuffer.cpp
${APHID_DIR}/shared/math/BaseBuffer.cpp
${APHID_DIR}/scan/CudaScan.cpp
${APHID_DIR}/reduction/CudaReduction.cpp)
set (CUSOURCES ${APHID_DIR}/radixsort/radixsort.cu
${APHID_DIR}/scan/scan.cu
${APHID_DIR}/reduction/cuReduceMax.cu
${APHID_DIR}/reduction/cuReduceMin.cu
${APHID_DIR}/reduction/cuReduceMinMax.cu
${APHID_DIR}/reduction/cuReduceMinMaxBox.cu
${APHID_DIR}/reduction/cuReduceSum.cu
${APHID_DIR}/taskQueue/SimpleQueue.cu
${APHID_DIR}/sahbvh/SahCompress.cu
${APHID_DIR}/sahbvh/SahDecompress.cu
${APHID_DIR}/sahbvh/SahSplit.cu)

CUDA_ADD_EXECUTABLE (testsplit ${HEADER_FILES} ${SOURC_FILES} ${CUSOURCES})

target_link_libraries(testsplit ${CUDA_CUDART_LIBRARY})

INCLUDE (${APHID_DIR}/cmake/boost_config.cmake)

target_link_libraries (testsplit ${Boost_SYSTEM_LIBRARY})
target_link_libraries (testsplit ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (testsplit ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (testsplit ${Boost_REGEX_LIBRARY})

target_link_libraries (testsplit ${APHID_FOUND_LIBRARY})
target_link_libraries (testsplit ${APHID_MATH_LIBRARY})
