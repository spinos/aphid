## http://www.cmake.org/cmake/help/v2.8.12/cmake.html#module:FindQt4
## http://www.cmake.org/cmake/help/v3.0/manual/cmake-qt.7.html#introduction

cmake_minimum_required (VERSION 2.8)
project (testadenium)
set (CMAKE_AUTOMOC ON)
set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_BUILD_TYPE RELEASE)

include_directories ("../shared"
"../cudadyn"
"../cudabvh"
"../cudanarrowphase"
"../radixsort"
"../scan" 
"../cudadyn"
"../reduction"
"../onebitsort"
"../sahbvh"
"../taskQueue"
"../hesperis")

set (HEADER_FILES "../shared/Base3DView.h" 
"window.h"
"glwidget.h"
"../shared/Morton3D.h"
"AdeniumWorld.h"
"AdeniumInterface.h")

set (SOURCE_FILES "../shared/Base3DView.cpp"
"../shared/CudaBase.cpp"
"../shared/CUDABuffer.cpp"
"../shared/CudaDbgLog.cpp"
"../scan/CudaScan.cpp"
"../reduction/CudaReduction.cpp"
"glwidget.cpp"
"window.cpp"
"main.cpp"
"../hesperis/HesperisFile.cpp"
"../sahbvh/BvhBuilder.cpp"
"../sahbvh/SahBuilder.cpp"
"../cudabvh/CudaLinearBvh.cpp"
"../cudabvh/TriangleSystem.cpp"
"../cudabvh/BvhTriangleSystem.cpp"
"../cudadyn/MassSystem.cpp"
"../cudadyn/CudaMassSystem.cpp"
"AdeniumWorld.cpp"
"AdeniumRender.cpp"
"AdeniumInterface.cpp")

if (WIN32)
set (SOURCE_FILES ${SOURCE_FILES} "../shared/gExtension.cpp")
endif ()

find_package (CUDA QUIET REQUIRED)

message ("cuda found ? " ${CUDA_FOUND})
message ("cuda version " ${CUDA_VERSION})
message ("cuda sdk root dir " ${CUDA_SDK_ROOT_DIR})
message ("cuda librarie " ${CUDA_CUDA_LIBRARY})
message ("cuda art librarie " ${CUDA_CUDART_LIBRARY})
message ("cuda blas librarie " ${CUDA_cublas_LIBRARY})
message ("cuda sparse librarie " ${CUDA_cusparse_LIBRARY})

set (CUSOURCES "../scan/scan.cu"
"../cudabvh/triangleSystem.cu"
"../cudabvh/tetrahedronSystem.cu"
"../reduction/cuReduceSum.cu"
"../reduction/cuReduceMin.cu"
"../reduction/cuReduceMax.cu"
"../reduction/cuReduceMinMax.cu"
"../reduction/cuReduceMinMaxBox.cu"
"../radixsort/radixsort.cu"
"../cudabvh/broadphase.cu"
"../cudanarrowphase/stripedModel.cu"
"../cudanarrowphase/matrix_math.cu"
"../cudanarrowphase/line_math.cu"
"../cudanarrowphase/triangle_math.cu"
"../cudanarrowphase/barycentric.cu"
"../cudanarrowphase/gjk_math.cu"
"../taskQueue/SimpleQueue.cu"
"../sahbvh/SahCompress.cu"
"../sahbvh/SahDecompress.cu"
"../sahbvh/SahSplit.cu"
"../sahbvh/BvhHash.cu"
"../sahbvh/BvhCost.cu"
"../sahbvh/BvhLazy.cu"
"AdeniumRender.cu")

if (WIN32)
set (CUDA_NVCC_FLAGS "--disable-warnings --ptxas-options=-v -arch sm_21")
else ()
set (CUDA_NVCC_FLAGS "--ptxas-options=-v -arch sm_11")
endif()

CUDA_ADD_EXECUTABLE (testadenium ${HEADER_FILES} ${SOURCE_FILES} ${CUSOURCES})

if (WIN32)
set (HDF5_INCLUDE_DIR "D:/usr/hdf5/include")
set (HDF5_LIBRARIES "D:/usr/hdf5/lib/hdf5.lib")
else ()
set (HDF5_INCLUDE_DIR "/users/jianzhang/Library/hdf5/include")
set (HDF5_LIBRARIES "/users/jianzhang/Library/hdf5/lib/libhdf5.dylib")
endif ()

if (WIN32)
set (SZIP_INCLUDE_DIR "D:/usr/szip-2.1/src")
set (SZIP_LIBRARIES "D:/usr/szip-2.1/szip.lib")
include_directories (${SZIP_INCLUDE_DIR})
target_link_libraries (testadenium ${SZIP_LIBRARIES})
endif ()

include_directories (${HDF5_INCLUDE_DIR})
target_link_libraries (testadenium ${HDF5_LIBRARIES})

find_package (OpenGL REQUIRED)
target_link_libraries (testadenium ${OPENGL_LIBRARIES})

if (WIN32)
set (Aphid_Library "D:/aphid/lib/aphid.lib")
else ()
set (Aphid_Library "/users/jianzhang/aphid/lib/libaphid.a")
endif ()
target_link_libraries (testadenium ${Aphid_Library})

if (WIN32)
set (OpenEXR_Library "D:/usr/openEXR/lib/Half.lib"
"D:/usr/openEXR/lib/Iex.lib"
"D:/usr/openEXR/lib/IlmImf.lib"
"D:/usr/openEXR/lib/IlmThread.lib")
else ()
set (OpenEXR_Library "/usr/local/lib/libHalf.a"
"/usr/local/lib/libIex.a"
"/usr/local/lib/libIlmImf.a"
"/usr/local/lib/libIlmThread.a")
endif ()
target_link_libraries (testadenium ${OpenEXR_Library})

find_package (zlib REQUIRED)
target_link_libraries (testadenium ${ZLIB_LIBRARIES})

if(WIN32)
set (QT_QMAKE_EXECUTABLE "D:/usr/qt4.8.6/bin/qmake.exe")
endif()

find_package (Qt4 REQUIRED)
message ("qt opengl " ${QT_QTOPENGL_LIBRARY})
set (QT_USE_QTOPENGL ON)
target_link_libraries (testadenium Qt4::QtGui Qt4::QtOpenGL)

if(WIN32)
	set (Boost_INCLUDE_DIR "D:/usr/boost_1_51_0")
	set (Boost_LIBRARY_DIR "D:/usr/boost_1_51_0/stage/lib")
	set (Boost_USE_STATIC_LIBS ON)
	set (Boost_USE_MULTITHREADED ON)
	include_directories ("D:/usr/boost_1_51_0")
	find_package(Boost 1.51 COMPONENTS system filesystem date_time regex thread chrono)
else()

if(APPLE)
	set (Boost_INCLUDE_DIR "/users/jianzhang/Library/boost_1_55_0")
	set (Boost_LIBRARY_DIR "/users/jianzhang/Library/boost_1_55_0/stage/lib")
	include_directories ("/users/jianzhang/Library/boost_1_55_0")
	find_package(Boost 1.55 COMPONENTS system filesystem date_time regex thread)
	include_directories ("/usr/local/include")

else()
	set (Boost_INCLUDE_DIR "/home/OF3D/zhangjian/Documents/boost_1_51_0")
	set (Boost_USE_STATIC_LIBS ON)
	set (Boost_USE_MULTITHREADED ON)
	include_directories ("/home/OF3D/zhangjian/Documents/boost_1_51_0")
	find_package(Boost 1.51 COMPONENTS system filesystem date_time regex thread)	
endif()
	
endif()

message ("boost system is" ${Boost_SYSTEM_LIBRARY})
message ("boost date_time is" ${Boost_DATE_TIME_LIBRARY})
message ("boost regex is" ${Boost_REGEX_LIBRARY})
message ("boost filesystem is" ${Boost_FILESYSTEM_LIBRARY})
message ("boost thread is" ${Boost_THREAD_LIBRARY})

set (AttributeNoninline "__attribute__ ((noinline))")
add_definitions (-DBOOST_NOINLINE=${AttributeNoninline})

message (${Boost_LIBRARIES})
target_link_libraries(testadenium ${Boost_LIBRARIES})

