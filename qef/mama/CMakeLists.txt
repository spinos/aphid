cmake_minimum_required (VERSION 2.8)
project (qef)

set (CMAKE_BUILD_TYPE Release)

if(WIN32)
    add_definitions (-D_BOOL -DREQUIRE_IOSTREAM -DNDEBUG -DNT_PLUGIN)
else ()
add_definitions (-DLINUX -DUNIX -D_BOOL -DFUNCPROTO -D_GNU_SOURCE -DREQUIRE_IOSTREAM -DBits64_ -DGL_GLEXT_PROTOTYPES)
endif()

message (CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR})
if(WIN32)
    
else()
ADD_DEFINITIONS(-fPIC -m64 -O3 -pthread)
endif()

include_directories("${PROJECT_BINARY_DIR}")

set (APHID_DIR "D:/aphid")
message (" aphid is  " ${APHID_DIR} )

include_directories (${APHID_DIR}/shared
    ${APHID_DIR}/btree
    ${APHID_DIR}/kdtree
    ${APHID_DIR}/ntree
    ${APHID_DIR}/hesperis)

set (SOURCES  main.cpp
qefCmd.cpp
${APHID_DIR}/btree/HWorldGrid.cpp
${APHID_DIR}/ntree/NTreeIO.cpp
${APHID_DIR}/kdtree/HNTree.cpp
${APHID_DIR}/shared/HElemAsset.cpp
${APHID_DIR}/shared/AHelper.cpp
${APHID_DIR}/shared/ASearchHelper.cpp
${APHID_DIR}/shared/SHelper.cpp
)

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

if(WIN32)
set (Boost_INCLUDE_DIR d:/usr/boost_1_51_0)
set (Boost_LIBRARY_DIR d:/usr/boost_1_51_0/stage/lib)
	set (Boost_USE_STATIC_LIBS ON)
	set (Boost_USE_MULTITHREADED ON)
	include_directories ("D:/usr/boost_1_51_0")
find_package(Boost 1.51 COMPONENTS system filesystem date_time regex thread chrono)
else()
set (HOME "/home/OF3D/zhangjian")
set (BOOST_ROOT ${HOME}/Documents/boost_1_51_0)
set (BOOST_LIBRARYDIR ${BOOST_ROOT}/stage/lib)
SET( Boost_USE_STATIC_LIBS TRUE )
SET( Boost_USE_MULTITHREADED TRUE )
SET( Boost_NO_BOOST_CMAKE TRUE ) 
find_package(Boost 1.51 COMPONENTS system filesystem date_time regex thread chrono)
endif()

message ("boost system is " ${Boost_SYSTEM_LIBRARY})
message ("boost date_time is " ${Boost_DATE_TIME_LIBRARY})
message ("boost regex is " ${Boost_REGEX_LIBRARY})
message ("boost filesystem is " ${Boost_FILESYSTEM_LIBRARY})
message ("boost thread is " ${Boost_THREAD_LIBRARY})
message ("boost chrono is " ${Boost_CHRONO_LIBRARY})

include_directories (${BOOST_ROOT})

if(WIN32)
else()
set (LINK_FLAGS ${LINK_FLAGS} -shared -fPIC -Wl)
endif()

if(WIN32)
set (MAYA_LOCATION_2014 "d:/usr/Maya2014")
set (MAYA_LOCATION_2015 "d:/usr/Maya2015")
set (MAYA_LOCATION_2016 "d:/usr/Maya2016")
else()
set (MAYA_LOCATION "/usr/autodesk/maya")
endif()

SET (MAYA_INCLUDE_PATH_2014 ${MAYA_LOCATION_2014}/include)
SET (MAYA_LIB_PATH_2014 ${MAYA_LOCATION_2014}/lib) 

SET (MAYA_INCLUDE_PATH_2015 ${MAYA_LOCATION_2015}/include)
SET (MAYA_LIB_PATH_2015 ${MAYA_LOCATION_2015}/lib)

SET (MAYA_INCLUDE_PATH_2016 ${MAYA_LOCATION_2016}/include)
SET (MAYA_LIB_PATH_2016 ${MAYA_LOCATION_2016}/lib) 

ADD_LIBRARY (qef2014 MODULE ${SOURCES})

target_link_libraries (qef2014 ${HDF5_LIBRARIES})
target_link_libraries (qef2014 ${ZLIB_LIBRARIES})
target_link_libraries (qef2014 ${SZIP_LIBRARY})

if(WIN32)
set_target_properties(qef2014 PROPERTIES SUFFIX .mll)
set(CMAKE_MODULE_LINKER_FLAGS "/export:initializePlugin /export:uninitializePlugin")
endif()
target_include_directories (qef2014 PRIVATE ${MAYA_INCLUDE_PATH_2014})

SET (APHID_LIBRARIES ${APHID_DIR}/lib/aphidMath.lib
${APHID_DIR}/lib/aphidGeom.lib
${APHID_DIR}/lib/aphidSdb.lib
${APHID_DIR}/lib/aphidH5.lib)

if(WIN32)
target_link_libraries (qef2014 ${APHID_LIBRARIES})
else()
endif()

target_link_libraries (qef2014 ${Boost_SYSTEM_LIBRARY})
target_link_libraries (qef2014 ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (qef2014 ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (qef2014 ${Boost_REGEX_LIBRARY})
target_link_libraries (qef2014 ${Boost_THREAD_LIBRARY})
target_link_libraries (qef2014 ${Boost_CHRONO_LIBRARY})

if(WIN32)
target_link_libraries (qef2014 ${MAYA_LIB_PATH_2014}/Foundation.lib
${MAYA_LIB_PATH_2014}/OpenMaya.lib
${MAYA_LIB_PATH_2014}/OpenMayaUI.lib
${MAYA_LIB_PATH_2014}/OpenMayaAnim.lib)
else()
target_link_libraries (qef2014 ${MAYA_LIB_PATH_2014}/libFoundation.so
${MAYA_LIB_PATH_2014}/libOpenMaya.so
${MAYA_LIB_PATH_2014}/libOpenMayaUI.so
${MAYA_LIB_PATH_2014}/libOpenMayaAnim.so)
endif()

ADD_LIBRARY (qef2015 MODULE ${SOURCES})
if(WIN32)
set_target_properties(qef2015 PROPERTIES SUFFIX .mll)
set(CMAKE_MODULE_LINKER_FLAGS "/export:initializePlugin /export:uninitializePlugin")
endif()
target_include_directories (qef2015 PRIVATE ${MAYA_INCLUDE_PATH_2015})

if(WIN32)
target_link_libraries (qef2015 ${APHID_LIBRARIES})
else()
endif()

target_link_libraries (qef2015 ${Boost_SYSTEM_LIBRARY})
target_link_libraries (qef2015 ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (qef2015 ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (qef2015 ${Boost_REGEX_LIBRARY})
target_link_libraries (qef2015 ${Boost_THREAD_LIBRARY})
target_link_libraries (qef2015 ${Boost_CHRONO_LIBRARY})

if(WIN32)
target_link_libraries (qef2015 ${MAYA_LIB_PATH_2015}/Foundation.lib
${MAYA_LIB_PATH_2015}/OpenMaya.lib
${MAYA_LIB_PATH_2015}/OpenMayaUI.lib
${MAYA_LIB_PATH_2015}/OpenMayaAnim.lib)
else()
target_link_libraries (qef2015 ${MAYA_LIB_PATH_2015}/libFoundation.so
${MAYA_LIB_PATH_2015}/libOpenMaya.so
${MAYA_LIB_PATH_2015}/libOpenMayaUI.so
${MAYA_LIB_PATH_2015}/libOpenMayaAnim.so)
endif()

target_link_libraries (qef2015 ${HDF5_LIBRARIES})
target_link_libraries (qef2015 ${ZLIB_LIBRARIES})
target_link_libraries (qef2015 ${SZIP_LIBRARY})

ADD_LIBRARY (qef2016 MODULE ${SOURCES})
if(WIN32)
set_target_properties(qef2016 PROPERTIES SUFFIX .mll)
set(CMAKE_MODULE_LINKER_FLAGS "/export:initializePlugin /export:uninitializePlugin")
endif()
target_include_directories (qef2016 PRIVATE ${MAYA_INCLUDE_PATH_2016})

if(WIN32)
target_link_libraries (qef2016 ${APHID_LIBRARIES})
else()
endif()

target_link_libraries (qef2016 ${Boost_SYSTEM_LIBRARY})
target_link_libraries (qef2016 ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (qef2016 ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (qef2016 ${Boost_REGEX_LIBRARY})
target_link_libraries (qef2016 ${Boost_THREAD_LIBRARY})
target_link_libraries (qef2016 ${Boost_CHRONO_LIBRARY})

if(WIN32)
target_link_libraries (qef2016 ${MAYA_LIB_PATH_2016}/Foundation.lib
${MAYA_LIB_PATH_2016}/OpenMaya.lib
${MAYA_LIB_PATH_2016}/OpenMayaUI.lib
${MAYA_LIB_PATH_2016}/OpenMayaAnim.lib)
else()
target_link_libraries (qef2016 ${MAYA_LIB_PATH_2016}/libFoundation.so
${MAYA_LIB_PATH_2016}/libOpenMaya.so
${MAYA_LIB_PATH_2016}/libOpenMayaUI.so
${MAYA_LIB_PATH_2016}/libOpenMayaAnim.so)
endif()

target_link_libraries (qef2016 ${HDF5_LIBRARIES})
target_link_libraries (qef2016 ${ZLIB_LIBRARIES})
target_link_libraries (qef2016 ${SZIP_LIBRARY})

