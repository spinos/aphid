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
    ${APHID_DIR}/rhizoid
    ${APHID_DIR}/btree
    ${APHID_DIR}/kdtree
    ${APHID_DIR}/hesperis
    ${APHID_DIR}/lapl)

set (SOURCES  main.cpp
qefCmd.cpp
${APHID_DIR}/shared/AHelper.cpp
${APHID_DIR}/shared/ASearchHelper.cpp
${APHID_DIR}/shared/SHelper.cpp
)

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
if(WIN32)
set_target_properties(qef2014 PROPERTIES SUFFIX .mll)
set(CMAKE_MODULE_LINKER_FLAGS "/export:initializePlugin /export:uninitializePlugin")
endif()
target_include_directories (qef2014 PRIVATE ${MAYA_INCLUDE_PATH_2014})

if(WIN32)
target_link_libraries (qef2014 ${APHID_DIR}/lib/aphidMath.lib)
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

if(WIN32)
target_link_libraries (qef2014 ${APHID_DIR}/lib/aphidMath.lib)
else()
endif()

ADD_LIBRARY (qef2015 MODULE ${SOURCES})
if(WIN32)
set_target_properties(qef2015 PROPERTIES SUFFIX .mll)
set(CMAKE_MODULE_LINKER_FLAGS "/export:initializePlugin /export:uninitializePlugin")
endif()
target_include_directories (qef2015 PRIVATE ${MAYA_INCLUDE_PATH_2015})

if(WIN32)
target_link_libraries (qef2015 ${APHID_DIR}/lib/aphidMath.lib)
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

if(WIN32)
target_link_libraries (qef2015 ${APHID_DIR}/lib/aphidMath.lib)
else()
endif()

ADD_LIBRARY (qef2016 MODULE ${SOURCES})
if(WIN32)
set_target_properties(qef2016 PROPERTIES SUFFIX .mll)
set(CMAKE_MODULE_LINKER_FLAGS "/export:initializePlugin /export:uninitializePlugin")
endif()
target_include_directories (qef2016 PRIVATE ${MAYA_INCLUDE_PATH_2016})

if(WIN32)
target_link_libraries (qef2016 ${APHID_DIR}/lib/aphidMath.lib)
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

if(WIN32)
target_link_libraries (qef2016 ${APHID_DIR}/lib/aphidMath.lib)
else()
endif()

