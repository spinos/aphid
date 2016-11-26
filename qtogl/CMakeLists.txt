## http://www.cmake.org/cmake/help/v2.8.12/cmake.html#module:FindQt4
## http://www.cmake.org/cmake/help/v3.0/manual/cmake-qt.7.html#introduction

cmake_minimum_required (VERSION 2.8)
project (qtgl)
set (CMAKE_AUTOMOC ON)
set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_BUILD_TYPE RELEASE)
include_directories (../shared
../bcc2d
../hesperis
../kdtree
../btree
../ntree)

if (WIN32)
SET (APHID_DIR "D:/aphid")
elseif (APPLE)
SET (APHID_DIR "/Users/jianzhang/aphid")
else()
SET (APHID_DIR "~/aphid")
endif ()

if (WIN32)
	set (APHID_MATH_LIBRARY ${APHID_DIR}/lib/aphidMath.lib)
    set (APHID_GEOM_LIBRARY ${APHID_DIR}/lib/aphidGeom.lib)
    set (APHID_H5_LIBRARY ${APHID_DIR}/lib/aphidH5.lib)
    set (APHID_UI_LIBRARY ${APHID_DIR}/lib/aphidUI.lib)
    set (APHID_KD_LIBRARY ${APHID_DIR}/lib/aphidGraph.lib)
    set (APHID_SDB_LIBRARY ${APHID_DIR}/lib/aphidKd.lib)
    set (APHID_GRAPH_LIBRARY ${APHID_DIR}/lib/aphidUI.lib)
elseif (APPLE)
    set (APHID_MATH_LIBRARY ${APHID_DIR}/lib/libaphidMath.a)
    set (APHID_GEOM_LIBRARY ${APHID_DIR}/lib/libaphidGeom.a)
    set (APHID_H5_LIBRARY ${APHID_DIR}/lib/libaphidH5.a)
    set (APHID_UI_LIBRARY ${APHID_DIR}/lib/libaphidUI.a)
    set (APHID_KD_LIBRARY ${APHID_DIR}/lib/libaphidKd.a)
    set (APHID_SDB_LIBRARY ${APHID_DIR}/lib/libaphidSdb.a)
    set (APHID_GRAPH_LIBRARY ${APHID_DIR}/lib/libaphidGraph.a)
ELSE ()
    SET (APHID_MATH_LIBRARY ${APHID_DIR}/lib/libaphidMath.so)
    SET (APHID_GEOM_LIBRARY ${APHID_DIR}/lib/libaphidGeom.so)
    SET (APHID_H5_LIBRARY ${APHID_DIR}/lib/libaphidH5.so)
    SET (APHID_UI_LIBRARY ${APHID_DIR}/lib/libaphidUI.so)
    SET (APHID_KD_LIBRARY ${APHID_DIR}/lib/libaphidKd.so)
    SET (APHID_SDB_LIBRARY ${APHID_DIR}/lib/libaphidSdb.so)
    SET (APHID_GRAPH_LIBRARY ${APHID_DIR}/lib/libaphidGraph.so)
endif ()

message ("aphid math is " ${APHID_MATH_LIBRARY})
message ("aphid geom is " ${APHID_GEOM_LIBRARY})
message ("aphid h5 is " ${APHID_H5_LIBRARY})
message ("aphid ui is " ${APHID_UI_LIBRARY})
message ("aphid kd is " ${APHID_KD_LIBRARY})
message ("aphid sdb is " ${APHID_SDB_LIBRARY})
message ("aphid graph is " ${APHID_GRAPH_LIBRARY})


if(WIN32)
	set (Boost_INCLUDE_DIR "D:/usr/boost_1_51_0")
	set (Boost_LIBRARY_DIR "D:/usr/boost_1_51_0/stage/lib")
	set (Boost_USE_STATIC_LIBS ON)
	set (Boost_USE_MULTITHREADED ON)
	include_directories ("D:/usr/boost_1_51_0")
	find_package(Boost 1.51 COMPONENTS system filesystem date_time regex thread chrono)

elseif(APPLE)
	set (Boost_INCLUDE_DIR "/users/jianzhang/Library/boost_1_55_0")
	set (Boost_LIBRARY_DIR "/users/jianzhang/Library/boost_1_55_0/stage/lib")
	include_directories ("/users/jianzhang/Library/boost_1_55_0")
	find_package(Boost 1.55 COMPONENTS system filesystem date_time regex thread chrono)
	include_directories ("/usr/local/include")

else()
    include_directories ("~/Public/boost_1_51_0")
    SET (BOOST_ROOT "~/Public/boost_1_51_0")
	find_package(Boost 1.51 COMPONENTS system filesystem date_time regex thread chrono zlib)	
endif()

message ("boost system is" ${Boost_SYSTEM_LIBRARY})
message ("boost date_time is" ${Boost_DATE_TIME_LIBRARY})
message ("boost regex is" ${Boost_REGEX_LIBRARY})
message ("boost filesystem is" ${Boost_FILESYSTEM_LIBRARY})
message ("boost thread is" ${Boost_THREAD_LIBRARY})

find_package (OpenGL REQUIRED)
message ("opengl is " ${OPENGL_LIBRARIES})

set (INST_HEADER_FILES ../shared/Base3DView.h 
instwindow.h
instwidget.h)

set (INST_SOURCE_FILES ../shared/Base3DView.cpp
instwidget.cpp
instwindow.cpp
instmain.cpp)

if (WIN32)
set (INST_SOURCE_FILES ${INST_SOURCE_FILES} "../shared/gExtension.cpp")
endif ()

add_executable (instdr ${INST_HEADER_FILES} ${INST_SOURCE_FILES})

if(WIN32)
set (QT_QMAKE_EXECUTABLE "D:/usr/qt4.8.6/bin/qmake.exe")
endif()

find_package (Qt4 REQUIRED)
message ("qt opengl " ${QT_QTOPENGL_LIBRARY})
set (QT_USE_QTOPENGL ON)
target_link_libraries (instdr Qt4::QtGui Qt4::QtOpenGL)

target_link_libraries (instdr ${OPENGL_LIBRARIES})

target_link_libraries (instdr ${Boost_SYSTEM_LIBRARY})
target_link_libraries (instdr ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (instdr ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (instdr ${Boost_REGEX_LIBRARY})
target_link_libraries (instdr ${Boost_THREAD_LIBRARY})
target_link_libraries (instdr ${Boost_CHRONO_LIBRARY})

set (AttributeNoninline "__attribute__ ((noinline))")
add_definitions (-DBOOST_NOINLINE=${AttributeNoninline})

target_link_libraries (instdr ${APHID_MATH_LIBRARY})
target_link_libraries (instdr ${APHID_GEOM_LIBRARY})
target_link_libraries (instdr ${APHID_UI_LIBRARY})

SET (EXECUTABLE_OUTPUT_PATH ../bin)

