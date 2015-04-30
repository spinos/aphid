## http://www.cmake.org/cmake/help/v2.8.12/cmake.html#module:FindCUDA

cmake_minimum_required (VERSION 2.8)
project (testcudacg)
# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)

find_package (CUDA QUIET REQUIRED)

message ("cuda found ? " ${CUDA_FOUND})
message ("cuda version " ${CUDA_VERSION})
message ("cuda sdk root dir " ${CUDA_SDK_ROOT_DIR})

include_directories (${CUDA_TOOLKIT_INCLUDE} ${CUDA_SDK_ROOT_DIR}/common/inc
"../radixsort"
"../cudabvh"
"../shared")

set (HEADER_FILES "../shared/Morton3D.h"
"../cudabvh/CudaReduction.h")
set (SOURC_FILES "main.cpp"
"../shared/CudaBase.cpp"
"../shared/CUDABuffer.cpp"
"../shared/BaseBuffer.cpp"
"../cudabvh/CudaReduction.cpp")
set (CUSOURCES "../radixsort/radixsort.cu"
"../cudabvh/reduceRange.cu"
"../cudabvh/cuReduceFlt2.cu"
"../cudabvh/cuReduceFlt3.cu"
"../cudabvh/cuReduceFlt4.cu"
"../cudabvh/cuReduceSum.cu")

## add_executable (testcudacg ${HEADER_FILES} main.cpp)
set (CUDA_NVCC_FLAGS "--ptxas-options=-v -arch sm_20")

CUDA_ADD_EXECUTABLE (testcudacg ${HEADER_FILES} ${SOURC_FILES} ${CUSOURCES})


target_link_libraries(testcudacg ${CUDA_CUDART_LIBRARY})
target_link_libraries(testcudacg ${CUDA_cublas_LIBRARY})
target_link_libraries(testcudacg ${CUDA_cusparse_LIBRARY})

if(WIN32)
	set (Boost_INCLUDE_DIR "D:/usr/boost_1_51_0")
	set (Boost_LIBRARY_DIR "D:/usr/boost_1_51_0/stage/lib")
	set (Boost_USE_STATIC_LIBS ON)
	set (Boost_USE_MULTITHREADED ON)
	include_directories ("D:/usr/boost_1_51_0")
	find_package(Boost 1.51 COMPONENTS system filesystem date_time regex)
else()

if(APPLE)
	set (Boost_INCLUDE_DIR "/users/jianzhang/Library/boost_1_55_0")
	set (Boost_LIBRARY_DIR "/users/jianzhang/Library/boost_1_55_0/stage/lib")
	include_directories ("/users/jianzhang/Library/boost_1_55_0")
	find_package(Boost 1.55 COMPONENTS system filesystem date_time regex)
	include_directories ("/usr/local/include")

else()
	set (Boost_INCLUDE_DIR "/home/OF3D/zhangjian/Documents/boost_1_51_0")
	set (Boost_USE_STATIC_LIBS ON)
	set (Boost_USE_MULTITHREADED ON)
	include_directories ("/home/OF3D/zhangjian/Documents/boost_1_51_0")
	find_package(Boost 1.51 COMPONENTS system filesystem date_time regex)	
endif()
	
endif()

message ("boost system is" ${Boost_SYSTEM_LIBRARY})
message ("boost date_time is" ${Boost_DATE_TIME_LIBRARY})
message ("boost regex is" ${Boost_REGEX_LIBRARY})
message ("boost filesystem is" ${Boost_FILESYSTEM_LIBRARY})

target_link_libraries (testcudacg ${Boost_SYSTEM_LIBRARY})
target_link_libraries (testcudacg ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (testcudacg ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (testcudacg ${Boost_REGEX_LIBRARY})


