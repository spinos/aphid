cmake_minimum_required (VERSION 2.8)
project (testla)
# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)
set (CMAKE_BUILD_TYPE RELEASE)
set (HEADER_FILES ../shared/Morton3D.h
clapackTempl.h
linearMath.h
regr.h)
include_directories ( ../shared ../kdtree)
add_executable (testla ${HEADER_FILES} testlapack.cpp)

if(WIN32)
    include_directories (D:/usr/CLAPACK/include)
    set(CBLAS_LIBRARY D:/usr/CLAPACK/lib/blas.lib)
	set(CLAPACK_LIBRARY D:/usr/CLAPACK/lib/lapack.lib)
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
