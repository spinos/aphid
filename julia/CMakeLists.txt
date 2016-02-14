cmake_minimum_required (VERSION 2.8)
project (julia)
SET (CMAKE_BUILD_TYPE Release)
include_directories ("." "../shared" "../btree")

set (HEADER_FILES ../btree/HWorldGrid.h
../btree/HInnerGrid.h
)

set (SOURCE_FILES Parameter.cpp 
QuatJulia.cpp
main.cpp)

add_executable(julia ${HEADER_FILES} ${SOURCE_FILES})

if (WIN32)
	IF (EXISTS "C:/Program Files/HDF5/cmake/hdf5")
# location of configure file FindHDF5.cmake
		SET (HDF5_DIR "C:/Program Files/HDF5/cmake/hdf5")
	ENDIF ()
endif ()

FIND_PACKAGE (HDF5 REQUIRED)

IF (WIN32)
SET (HDF5_LIBRARIES "C:/Program Files/HDF5/lib/libhdf5.lib")
ENDIF ()

MESSAGE (" hdf5 version is " ${HDF5_VERSION} )
MESSAGE (" hdf5 include is " ${HDF5_INCLUDE_DIRS} )
MESSAGE (" hdf5 library is " ${HDF5_LIBRARIES} )

include_directories (${HDF5_INCLUDE_DIRS})
target_link_libraries (julia ${HDF5_LIBRARIES})

if(WIN32)
SET (ZLIB_DIR "C:/Program Files/ZLIB/cmake/ZLIB")
SET (ZLIB_INCLUDE_DIR "C:/Program Files/ZLIB/include")
SET (ZLIB_LIBRARY "C:/Program Files/ZLIB/lib/libzlib.lib")
endif()

FIND_PACKAGE (ZLIB REQUIRED)
MESSAGE (" zlib version major is " ${ZLIB_VERSION_MAJOR})
MESSAGE (" zlib version minor is " ${ZLIB_VERSION_MINOR})
MESSAGE (" zlib include is " ${ZLIB_INCLUDE_DIR})
MESSAGE (" zlib library is " ${ZLIB_LIBRARY})

target_link_libraries (julia ${ZLIB_LIBRARIES})

IF (WIN32)
set (SZIP_DIR "C:/Program Files/SZIP/cmake/SZIP")
set (SZIP_INCLUDE_DIR "C:/Program Files/SZIP/include")
set (SZIP_LIBRARY "C:/Program Files/SZIP/lib/libszip.lib")

ELSEIF (APPLE)
set (SZIP_DIR "/usr/local/share/cmake/SZIP")
set (SZIP_INCLUDE_DIR "/usr/local/include")
set (SZIP_LIBRARY "/usr/local/lib/libszip.a")

ENDIF ()

FIND_PACKAGE (SZIP REQUIRED)
MESSAGE (" szip version major is " ${SZIP_VERSION_MAJOR})
MESSAGE (" szip version minor is " ${SZIP_VERSION_MINOR})
MESSAGE (" szip include is " ${SZIP_INCLUDE_DIR})
MESSAGE (" szip library is " ${SZIP_LIBRARY})

target_link_libraries (julia ${SZIP_LIBRARY})

if(WIN32)
	IF (EXISTS "D:/usr/boost_1_51_0")
		SET (Boost_INCLUDE_DIR "D:/usr/boost_1_51_0")
	
	ENDIF ()
	set (Boost_USE_STATIC_LIBS ON)
	set (Boost_USE_MULTITHREADED ON)
	
ELSEIF (APPLE)
	IF (EXISTS "/users/jianzhang/Library/boost_1_55_0")
		SET (Boost_INCLUDE_DIR "/users/jianzhang/Library/boost_1_55_0")
	ELSEIF (EXISTS "/users/jollyroger/Library/boost_1_43_0")
		SET (Boost_INCLUDE_DIR "/users/jollyroger/Library/boost_1_43_0")
	ENDIF ()
	
endif()

FIND_PACKAGE (Boost REQUIRED COMPONENTS system filesystem date_time regex)	
	
MESSAGE (" boost version is " ${Boost_VERSION})
MESSAGE (" boost include is " ${Boost_INCLUDE_DIR} )
MESSAGE (" boost system is " ${Boost_SYSTEM_LIBRARY})
MESSAGE (" boost date_time is " ${Boost_DATE_TIME_LIBRARY})
MESSAGE (" boost regex is " ${Boost_REGEX_LIBRARY})
MESSAGE (" boost filesystem is " ${Boost_FILESYSTEM_LIBRARY})

include_directories (${Boost_INCLUDE_DIR})
	
target_link_libraries (julia ${Boost_SYSTEM_LIBRARY})
target_link_libraries (julia ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (julia ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (julia ${Boost_REGEX_LIBRARY})

if (WIN32)
set (Aphid_Library "D:/aphid/lib/aphid.lib")
else ()
	IF (EXISTS "/users/jianzhang/aphid/lib/libaphid.a")
		set (Aphid_Library "/users/jianzhang/aphid/lib/libaphid.a")
	ELSEIF (EXISTS "/users/jollyroger/aphid/lib/libaphid.a")
		set (Aphid_Library "/users/jollyroger/aphid/lib/libaphid.a")
	ENDIF ()
endif ()

target_link_libraries (julia ${Aphid_Library})

