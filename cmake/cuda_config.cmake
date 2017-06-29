MESSAGE (" find CUDA ")

find_package (CUDA QUIET)

IF (${CUDA_FOUND})

message ("cuda found ? " ${CUDA_FOUND})
message ("cuda version " ${CUDA_VERSION})
message ("cuda sdk root " ${CUDA_SDK_ROOT_DIR})
message ("cuda librarie " ${CUDA_CUDA_LIBRARY})
message ("cuda art librarie " ${CUDA_CUDART_LIBRARY})
message ("cuda blas librarie " ${CUDA_cublas_LIBRARY})
message ("cuda sparse librarie " ${CUDA_cusparse_LIBRARY})
message ("cuda toolkit include " ${CUDA_TOOLKIT_INCLUDE})

find_library (CUDA_CUT_LIBRARY NAMES libcutil_${CMAKE_SYSTEM_PROCESSOR}
	HINTS ${CUDA_SDK_ROOT_DIR}/lib
	DOC "Location of cutil library")

message ("cuda cut library " ${CUDA_CUT_LIBRARY})

if (WIN32)
set (CUDA_NVCC_FLAGS "--disable-warnings --ptxas-options=-v -arch sm_21")
else ()
set (CUDA_NVCC_FLAGS "--ptxas-options=-v -arch sm_11")
endif()

include_directories (${CUDA_TOOLKIT_INCLUDE} ${CUDA_SDK_ROOT_DIR}/common/inc)

ENDIF ()
