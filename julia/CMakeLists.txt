cmake_minimum_required (VERSION 2.8)
project (julia)
 
include_directories ("." "../shared" "../btree")

set (HEADER_FILES ../btree/HWorldGrid.h
../btree/HInnerGrid.h
)

set (SOURCE_FILES Parameter.cpp 
QuatJulia.cpp
main.cpp)

add_executable(julia ${HEADER_FILES} ${SOURCE_FILES})

if (WIN32)
	IF (EXISTS "D:/usr/hdf5/include")
		SET (HDF5_INCLUDE_DIR "D:/usr/hdf5/include")
	ENDIF ()

elseif (APPLE)
	IF (EXISTS "/users/jianzhang/Library/hdf5/include")
		SET (HDF5_INCLUDE_DIR "/users/jianzhang/Library/hdf5/include")
	ENDIF ()
	
endif ()

FIND_PACKAGE (HDF5 REQUIRED)
	
MESSAGE (" hdf5 version is " ${HDF5_VERSION} )
MESSAGE (" hdf5 include is " ${HDF5_INCLUDE_DIR} )
MESSAGE (" hdf5 library is " ${HDF5_LIBRARIES} )

if(WIN32)
set (ZLIB_ROOT "D:/usr/zlib-1.2.5")
target_link_libraries (julia "D:/usr/hdf5/lib/zlib.lib")
endif()


include_directories (${HDF5_INCLUDE_DIR})
target_link_libraries (julia ${HDF5_LIBRARIES})

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

