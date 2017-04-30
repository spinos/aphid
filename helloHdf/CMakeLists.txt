cmake_minimum_required (VERSION 2.8)
project (helloh5)
set (CMAKE_AUTOMOC ON)
set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_BUILD_TYPE RELEASE)
# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)
set (HEADER_FILES "../shared/Morton3D.h")
include_directories ("../shared")

add_executable (hdata ${HEADER_FILES} Holder.cpp main.cpp)

SET (INSTALLED_HDF5 OFF)
if (WIN32)
	IF (EXISTS "C:/Program Files/HDF5/cmake/hdf5")
# location of configure file FindHDF5.cmake
		SET (HDF5_DIR "C:/Program Files/HDF5/cmake/hdf5")
        SET (INSTALLED_HDF5 ON)
    ELSE (EXISTS "D:/usr/hdf5")
        SET (HDF5_INCLUDE_DIRS "D:/usr/hdf5/include")
        SET (HDF5_LIBRARIES "D:/usr/hdf5/lib/hdf5.lib")
	ENDIF ()
ELSE ()
	SET (INSTALLED_HDF5 ON)
endif ()

IF (INSTALLED_HDF5)
FIND_PACKAGE (HDF5 REQUIRED)
IF (WIN32)
SET (HDF5_LIBRARIES "C:/Program Files/HDF5/lib/libhdf5.lib")	
ENDIF ()
ENDIF ()

MESSAGE (" hdf5 version is " ${HDF5_VERSION} )
MESSAGE (" hdf5 include is " ${HDF5_INCLUDE_DIRS} )
MESSAGE (" hdf5 library is " ${HDF5_LIBRARIES} )

include_directories (${HDF5_INCLUDE_DIRS})
target_link_libraries (hdata ${HDF5_LIBRARIES})

if(WIN32)
SET (ZLIB_DIR "D:/usr/zlib-1.2.5")
SET (ZLIB_INCLUDE_DIR "D:/usr/zlib-1.2.5")
SET (ZLIB_LIBRARY "D:/usr/zlib-1.2.5/zlib.lib")
endif()

FIND_PACKAGE (ZLIB REQUIRED)
MESSAGE (" zlib version major is " ${ZLIB_VERSION_MAJOR})
MESSAGE (" zlib version minor is " ${ZLIB_VERSION_MINOR})
MESSAGE (" zlib include is " ${ZLIB_INCLUDE_DIR})
MESSAGE (" zlib library is " ${ZLIB_LIBRARY})

target_link_libraries (hdata ${ZLIB_LIBRARIES})

IF (WIN32)
set (SZIP_DIR "C:/Program Files/SZIP/share/cmake/SZIP")
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

target_link_libraries (hdata ${SZIP_LIBRARY})

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
	
target_link_libraries (hdata ${Boost_SYSTEM_LIBRARY})
target_link_libraries (hdata ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (hdata ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (hdata ${Boost_REGEX_LIBRARY})

if (WIN32)
SET (Aphid_Math_Library D:/aphid/lib/aphidMath.lib)
SET (Aphid_Sdb_Library D:/aphid/lib/aphidSdb.lib)
SET (Aphid_H5_Library D:/aphid/lib/aphidH5.lib)
else ()
	IF (EXISTS "/users/jianzhang/aphid/lib/libaphidMath.a")
		SET (Aphid_Math_Library /users/jianzhang/aphid/lib/libaphidMath.a) 
		SET (Aphid_Sdb_Library /users/jianzhang/aphid/lib/libaphidSdb.a) 
		SET (Aphid_H5_Library /users/jianzhang/aphid/lib/libaphidH5.a )
		SET (Aphid_Geom_Library /users/jianzhang/aphid/lib/libaphidGeom.a )
		SET (Aphid_Cu_Library /users/jianzhang/aphid/lib/libaphidCu.a )
	ELSEIF (EXISTS "/users/jollyroger/aphid/lib/libaphidMath.a")
		set (Aphid_Library "/users/jollyroger/aphid/lib/libaphid.a")
	ENDIF ()
endif ()

MESSAGE (" aphid math library is " ${Aphid_Math_Library})
MESSAGE (" aphid sdb library is " ${Aphid_Sdb_Library})
MESSAGE (" aphid h5 library is " ${Aphid_H5_Library})
MESSAGE (" aphid geom library is " ${Aphid_Geom_Library})
MESSAGE (" aphid cu library is " ${Aphid_Cu_Library})

target_link_libraries (hdata ${Aphid_Math_Library})
target_link_libraries (hdata ${Aphid_Sdb_Library})
target_link_libraries (hdata ${Aphid_H5_Library})

add_executable (checkIntegrity checkIntegrityMain.cpp)

target_link_libraries (checkIntegrity ${HDF5_LIBRARIES})
target_link_libraries (checkIntegrity ${ZLIB_LIBRARIES})
target_link_libraries (checkIntegrity ${SZIP_LIBRARY})
target_link_libraries (checkIntegrity ${Aphid_H5_Library})
target_link_libraries (checkIntegrity ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (checkIntegrity ${Boost_SYSTEM_LIBRARY})

add_executable (diagnose diagnoseMain.cpp)

target_link_libraries (diagnose ${HDF5_LIBRARIES})
target_link_libraries (diagnose ${ZLIB_LIBRARIES})
target_link_libraries (diagnose ${SZIP_LIBRARY})
target_link_libraries (diagnose ${Aphid_H5_Library})
target_link_libraries (diagnose ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (diagnose ${Boost_SYSTEM_LIBRARY})
target_link_libraries (diagnose ${Boost_REGEX_LIBRARY})
