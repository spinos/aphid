## http://www.cmake.org/cmake/help/v2.8.12/cmake.html#module:FindQt4
## http://www.cmake.org/cmake/help/v3.0/manual/cmake-qt.7.html#introduction

cmake_minimum_required (VERSION 2.8)
project (glstrudy)
set (CMAKE_AUTOMOC ON)
set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_BUILD_TYPE RELEASE)
include_directories (../shared)

set (INST_HEADER_FILES ../shared/Base3DView.h 
instwindow.h
instwidget.h)

set (INST_SOURCE_FILES ../shared/Base3DView.cpp
instwidget.cpp
instwindow.cpp
instmain.cpp)

if (WIN32)
SET (APHID_DIR "D:/aphid")
elseif (APPLE)
SET (APHID_DIR "/Users/jianzhang/aphid")
else()
SET (APHID_DIR "~/aphid")
endif ()

if(WIN32)
add_definitions (-DWIN32)
endif()

find_package (OpenGL REQUIRED)
message ("opengl is " ${OPENGL_LIBRARIES})

if (WIN32)
SET (APHID_FOUND_LIBRARY ${APHID_DIR}/lib/aphidFoundation.lib)
SET (APHID_MATH_LIBRARY ${APHID_DIR}/lib/aphidMath.lib)
    SET (APHID_GEOM_LIBRARY ${APHID_DIR}/lib/aphidGeom.lib)
    SET (APHID_H5_LIBRARY ${APHID_DIR}/lib/aphidH5.lib)
    SET (APHID_UI_LIBRARY ${APHID_DIR}/lib/aphidUI.lib)
    SET (APHID_KD_LIBRARY ${APHID_DIR}/lib/aphidKd.lib)
    SET (APHID_SDB_LIBRARY ${APHID_DIR}/lib/aphidSdb.lib)
    SET (APHID_GRAPH_LIBRARY ${APHID_DIR}/lib/aphidGraph.lib)
set (APHID_TOPO_LIBRARY ${APHID_DIR}/lib/aphidTopology.lib)
set (APHID_OGL_LIBRARY ${APHID_DIR}/lib/aphidOgl.lib)

elseif (APPLE)
SET (APHID_FOUND_LIBRARY ${APHID_DIR}/lib/libaphidFoundation.a)
    set (APHID_MATH_LIBRARY ${APHID_DIR}/lib/libaphidMath.a)
    set (APHID_GEOM_LIBRARY ${APHID_DIR}/lib/libaphidGeom.a)
    set (APHID_H5_LIBRARY ${APHID_DIR}/lib/libaphidH5.a)
    set (APHID_UI_LIBRARY ${APHID_DIR}/lib/libaphidUI.a)
    set (APHID_KD_LIBRARY ${APHID_DIR}/lib/libaphidKd.a)
    set (APHID_SDB_LIBRARY ${APHID_DIR}/lib/libaphidSdb.a)
    set (APHID_GRAPH_LIBRARY ${APHID_DIR}/lib/libaphidGraph.a)
set (APHID_TOPO_LIBRARY ${APHID_DIR}/lib/libaphidTopology.a)
set (APHID_OGL_LIBRARY ${APHID_DIR}/lib/libaphidOgl.a)

ELSE ()
SET (APHID_FOUND_LIBRARY ${APHID_DIR}/lib/libaphidFoundation.so)
    SET (APHID_MATH_LIBRARY ${APHID_DIR}/lib/libaphidMath.so)
    SET (APHID_GEOM_LIBRARY ${APHID_DIR}/lib/libaphidGeom.so)
    SET (APHID_H5_LIBRARY ${APHID_DIR}/lib/libaphidH5.so)
    SET (APHID_UI_LIBRARY ${APHID_DIR}/lib/libaphidUI.so)
    SET (APHID_KD_LIBRARY ${APHID_DIR}/lib/libaphidKd.so)
    SET (APHID_SDB_LIBRARY ${APHID_DIR}/lib/libaphidSdb.so)
    SET (APHID_GRAPH_LIBRARY ${APHID_DIR}/lib/libaphidGraph.so)
set (APHID_TOPO_LIBRARY ${APHID_DIR}/lib/libaphidGraph.so)
set (APHID_OGL_LIBRARY ${APHID_DIR}/lib/libaphidOgl.so)

endif ()

message ("aphid foundation is " ${APHID_FOUND_LIBRARY})
message ("aphid math is " ${APHID_MATH_LIBRARY})
message ("aphid geom is " ${APHID_GEOM_LIBRARY})
message ("aphid h5 is " ${APHID_H5_LIBRARY})
message ("aphid ui is " ${APHID_UI_LIBRARY})
message ("aphid kd is " ${APHID_KD_LIBRARY})
message ("aphid sdb is " ${APHID_SDB_LIBRARY})
message ("aphid graph is " ${APHID_GRAPH_LIBRARY})
message ("aphid topo is " ${APHID_TOPO_LIBRARY})
message ("aphid ogl is " ${APHID_OGL_LIBRARY})


if(WIN32)
set (QT_QMAKE_EXECUTABLE "D:/usr/qt4.8.6/bin/qmake.exe")
endif()

find_package (Qt4 REQUIRED)
message ("qt opengl is " ${QT_QTOPENGL_LIBRARY})
set (QT_USE_QTOPENGL ON)

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

message ("boost system is     " ${Boost_SYSTEM_LIBRARY})
message ("boost date_time is  " ${Boost_DATE_TIME_LIBRARY})
message ("boost regex is      " ${Boost_REGEX_LIBRARY})
message ("boost filesystem is " ${Boost_FILESYSTEM_LIBRARY})
message ("boost thread is     " ${Boost_THREAD_LIBRARY})

add_executable (psinst ${INST_HEADER_FILES} ${INST_SOURCE_FILES})

target_link_libraries (psinst Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (psinst ${OPENGL_LIBRARIES})
target_link_libraries (psinst ${APHID_FOUND_LIBRARY})
target_link_libraries (psinst ${APHID_MATH_LIBRARY})
target_link_libraries (psinst ${APHID_GEOM_LIBRARY})
target_link_libraries (psinst ${APHID_UI_LIBRARY})
target_link_libraries (psinst ${APHID_SDB_LIBRARY})
target_link_libraries (psinst ${APHID_TOPO_LIBRARY})
target_link_libraries (psinst ${APHID_OGL_LIBRARY})

target_link_libraries (psinst ${Boost_SYSTEM_LIBRARY})
target_link_libraries (psinst ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (psinst ${Boost_REGEX_LIBRARY})
target_link_libraries (psinst ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (psinst ${Boost_THREAD_LIBRARY})
target_link_libraries (psinst ${Boost_CHRONO_LIBRARY})

set (AttributeNoninline "__attribute__ ((noinline))")
add_definitions (-DBOOST_NOINLINE=${AttributeNoninline})

SET (EXECUTABLE_OUTPUT_PATH ../bin)

if(WIN32)
    include_directories (D:/usr/CLAPACK/include)
    set(BLAS_LIBRARY D:/usr/CLAPACK/lib/blas.lib)
	set(LAPACK_LIBRARY D:/usr/CLAPACK/lib/lapack.lib)
    set(F2C_LIBRARY D:/usr/CLAPACK/lib/libf2c.lib)

elseif(APPLE)
	set(F2C_LIBRARY libf2c.a)
	set(BLAS_LIBRARY libblas.a)
	set(LAPACK_LIBRARY liblapack.a)
else ()
include_directories (/usr/local/include)
	set(F2C_LIBRARY /usr/local/lib/libf2c.so)
    set(BLAS_LIBRARY /usr/local/lib/libblas.so)
	set(LAPACK_LIBRARY /usr/local/lib/liblapack.so)
	
endif()

message ("blas " ${BLAS_LIBRARY})
message ("lapack " ${LAPACK_LIBRARY})
message ("f2c " ${F2C_LIBRARY})

add_executable (testclique testClique.cpp)
target_link_libraries (testclique ${APHID_FOUND_LIBRARY})
target_link_libraries (testclique ${APHID_MATH_LIBRARY})
target_link_libraries (testclique ${APHID_GEOM_LIBRARY})
target_link_libraries (testclique ${APHID_TOPO_LIBRARY})
target_link_libraries (testclique ${APHID_SDB_LIBRARY})

target_link_libraries (testclique ${BLAS_LIBRARY})
target_link_libraries (testclique ${LAPACK_LIBRARY})
target_link_libraries (testclique ${F2C_LIBRARY})

target_link_libraries (testclique ${Boost_SYSTEM_LIBRARY})
target_link_libraries (testclique ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (testclique ${Boost_REGEX_LIBRARY})
target_link_libraries (testclique ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (testclique ${Boost_THREAD_LIBRARY})
target_link_libraries (testclique ${Boost_CHRONO_LIBRARY})

SET (GPDF_HEADER cactus.h)

SET (GPDF_SOURCE ../shared/Base3DView.cpp
../shared/BaseImageWidget.cpp
../shared/Plot1DWidget.cpp
gpdfwidget.cpp
gpdfwindow.cpp
gpdfxwidget.cpp
gpdfxdialog.cpp
gpdfmain.cpp)

add_executable (gpdeform ${GP_HEADER} ${GPDF_SOURCE})
target_link_libraries (gpdeform Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (gpdeform ${OPENGL_LIBRARIES})
target_link_libraries (gpdeform ${APHID_FOUND_LIBRARY})
target_link_libraries (gpdeform ${APHID_MATH_LIBRARY})
target_link_libraries (gpdeform ${APHID_GEOM_LIBRARY})
target_link_libraries (gpdeform ${APHID_UI_LIBRARY})
target_link_libraries (gpdeform ${APHID_SDB_LIBRARY})
target_link_libraries (gpdeform ${APHID_TOPO_LIBRARY})
target_link_libraries (gpdeform ${APHID_OGL_LIBRARY})

target_link_libraries (gpdeform ${BLAS_LIBRARY})
target_link_libraries (gpdeform ${LAPACK_LIBRARY})
target_link_libraries (gpdeform ${F2C_LIBRARY})

target_link_libraries (gpdeform ${Boost_DATE_TIME_LIBRARY})

add_subdirectory(rott)
add_subdirectory(naca4)
add_subdirectory(hermite)
add_subdirectory(pca)
