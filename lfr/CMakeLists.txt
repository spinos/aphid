cmake_minimum_required (VERSION 2.8)
project (testla)
# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)
set (CMAKE_AUTOMOC ON)
set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_BUILD_TYPE RELEASE)
set (HEADER_FILES ../shared/Morton3D.h
clapackTempl.h
linearMath.h
regr.h)
include_directories ( ../shared ../kdtree)
add_executable (testla ${HEADER_FILES} 
../shared/zEXRImage.cpp
LfWorld.cpp
LfThread.cpp
LfWidget.cpp
main.cpp)

add_definitions (-DOPENEXR_DLL)

if(WIN32)
    include_directories (D:/usr/CLAPACK/include)
    set(BLAS_LIBRARY D:/usr/CLAPACK/lib/blas.lib)
	set(LAPACK_LIBRARY D:/usr/CLAPACK/lib/lapack.lib)
    set(F2C_LIBRARY D:/usr/CLAPACK/lib/libf2c.lib)
    
endif()

if(APPLE)
	set(F2C_LIBRARY libf2c.a)
	set(BLAS_LIBRARY libblas.a)
	set(LAPACK_LIBRARY liblapack.a)
endif()

message ("blas " ${BLAS_LIBRARY})
message ("lapack " ${LAPACK_LIBRARY})
message ("f2c " ${F2C_LIBRARY})

target_link_libraries (testla ${F2C_LIBRARY})
target_link_libraries (testla ${BLAS_LIBRARY})
target_link_libraries (testla ${LAPACK_LIBRARY})

if (WIN32)
set (OpenEXR_INCLUDE_DIR D:/usr/openexr21/include)
set (OpenEXR_Library "D:/usr/openexr21/lib/Half.lib"
"D:/usr/openexr21/lib/Iex-2_1.lib"
"D:/usr/openexr21/lib/IlmImf-2_1.lib"
"D:/usr/openexr21/lib/IlmThread-2_1.lib")
else ()
set (OpenEXR_INCLUDE_DIR /usr/local/include/OpenEXR)
set (OpenEXR_Library /usr/local/lib/libHalf.a
 /usr/local/lib/libIex-2_1.dylib
 /usr/local/lib/libIlmImf-2_1.dylib
 /usr/local/lib/libIlmThread-2_1.dylib)
endif ()
include_directories (${OpenEXR_INCLUDE_DIR})
target_link_libraries (testla ${OpenEXR_Library})
message (" openexr " ${OpenEXR_Library})

find_package (zlib REQUIRED)
target_link_libraries (testla ${ZLIB_LIBRARIES})

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
target_link_libraries (testla ${SZIP_LIBRARIES})
endif ()

include_directories (${HDF5_INCLUDE_DIR})
target_link_libraries (testla ${HDF5_LIBRARIES})

if(WIN32)
set (QT_QMAKE_EXECUTABLE "D:/usr/qt4.8.6/bin/qmake.exe")
endif()

find_package (Qt4 REQUIRED)
target_link_libraries (testla Qt4::QtGui)
## message ("qt opengl " ${QT_QTOPENGL_LIBRARY})
## set (QT_USE_QTOPENGL ON)
## target_link_libraries (testla Qt4::QtGui Qt4::QtOpenGL)

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

message ("boost system " ${Boost_SYSTEM_LIBRARY})
message ("boost date_time " ${Boost_DATE_TIME_LIBRARY})
message ("boost regex " ${Boost_REGEX_LIBRARY})
message ("boost filesystem " ${Boost_FILESYSTEM_LIBRARY})
message ("boost thread " ${Boost_THREAD_LIBRARY})

set (AttributeNoninline "__attribute__ ((noinline))")
add_definitions (-DBOOST_NOINLINE=${AttributeNoninline})

target_link_libraries (testla ${Boost_SYSTEM_LIBRARY})
target_link_libraries (testla ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (testla ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (testla ${Boost_REGEX_LIBRARY})
target_link_libraries (testla ${Boost_THREAD_LIBRARY})
target_link_libraries (testla ${Boost_CHRONO_LIBRARY})

if (WIN32)
set (Aphid_Library "D:/aphid/lib/aphid.lib")
else ()
set (Aphid_Library "/users/jianzhang/aphid/lib/libaphid.a")
endif ()

target_link_libraries (testla ${Aphid_Library})


