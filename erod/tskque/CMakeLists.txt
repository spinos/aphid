INCLUDE (${APHID_DIR}/cmake/cuda_config.cmake)

include_directories (${APHID_DIR}/radixsort
${APHID_DIR}/reduction
${APHID_DIR}/cudabvh
${APHID_DIR}/scan
${APHID_DIR}/taskQueue)

set (HEADER_FILES ${APHID_DIR}/taskQueue/SimpleQueueInterface.h)
set (SOURC_FILES main.cpp
${APHID_DIR}/shared/CUDABuffer.cpp
${APHID_DIR}/shared/BaseLog.cpp
${APHID_DIR}/shared/CudaDbgLog.cpp
${APHID_DIR}/shared/math/BaseBuffer.cpp)
set (CUSOURCES ${APHID_DIR}/taskQueue/quickSort.cu
${APHID_DIR}/taskQueue/SimpleQueue.cu)

CUDA_ADD_EXECUTABLE (testqueue ${HEADER_FILES} ${SOURC_FILES} ${CUSOURCES})

target_link_libraries(testqueue ${CUDA_CUDART_LIBRARY})

INCLUDE (${APHID_DIR}/cmake/boost_config.cmake)

target_link_libraries (testqueue ${Boost_SYSTEM_LIBRARY})
target_link_libraries (testqueue ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (testqueue ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (testqueue ${Boost_REGEX_LIBRARY})

target_link_libraries (testqueue ${APHID_FOUND_LIBRARY})
target_link_libraries (testqueue ${APHID_MATH_LIBRARY})
target_link_libraries (testqueue ${APHID_CU_LIBRARY})
