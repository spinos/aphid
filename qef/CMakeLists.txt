cmake_minimum_required (VERSION 2.8)
project (testqef)
# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)
set (CMAKE_AUTOMOC ON)
set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_BUILD_TYPE RELEASE)
set (HEADER_FILES
../lfr/clapackTempl.h
../lfr/linearMath.h
../shared/KMeansClustering.h
../shared/QuadraticErrorFunction.h)
include_directories ( ../shared ../lfr ../btree)
add_executable (testqef ${HEADER_FILES} main.cpp)

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

target_link_libraries (testqef ${F2C_LIBRARY})
target_link_libraries (testqef ${BLAS_LIBRARY})
target_link_libraries (testqef ${LAPACK_LIBRARY})

if (WIN32)
set (Aphid_Library "D:/aphid/lib/aphid.lib")
else ()
set (Aphid_Library /users/jianzhang/aphid/lib/libaphidMath.a
/users/jianzhang/aphid/lib/libaphidSdb.a)
endif ()

target_link_libraries (testqef ${Aphid_Library})


