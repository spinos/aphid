## http://www.cmake.org/cmake/help/v2.8.12/cmake.html#module:FindQt4
## http://www.cmake.org/cmake/help/v3.0/manual/cmake-qt.7.html#introduction

cmake_minimum_required (VERSION 2.8)
project (glstrudy)
set (CMAKE_AUTOMOC ON)
set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_BUILD_TYPE RELEASE)
include_directories (../shared)

if (WIN32)
SET (APHID_DIR "D:/aphid")
elseif (APPLE)
IF (EXISTS /Users/jollyroger)
	SET (APPLE_HOME /Users/jollyroger)
ELSEIF (EXISTS /Users/jianzhang)
	SET (APPLE_HOME /Users/jianzhang)
ENDIF()

SET (APHID_DIR ${APPLE_HOME}/aphid)
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
set (APHID_TTG_LIBRARY ${APHID_DIR}/lib/aphidTtg.lib)
set (APHID_IMG_LIBRARY ${APHID_DIR}/lib/aphidImg.lib)

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
set (APHID_TTG_LIBRARY ${APHID_DIR}/lib/libaphidTtg.a)
set (APHID_IMG_LIBRARY ${APHID_DIR}/lib/libaphidImg.a)

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
set (APHID_TTG_LIBRARY ${APHID_DIR}/lib/libaphidTtg.so)
set (APHID_IMG_LIBRARY ${APHID_DIR}/lib/libaphidImg.so)

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
message ("aphid ttg is " ${APHID_TTG_LIBRARY})

SET (INSTALLED_HDF5 OFF)
if (WIN32)
	IF (EXISTS "C:/Program Files/HDF5/cmake/hdf5")
# location of configure file FindHDF5.cmake
		SET (HDF5_DIR "C:/Program Files/HDF5/cmake/hdf5")
        SET (INSTALLED_HDF5 ON)
    ELSEIF (EXISTS "D:/usr/hdf5")
        SET (HDF5_INCLUDE_DIRS "D:/usr/hdf5/include")
        SET (HDF5_LIBRARIES "D:/usr/hdf5/lib/hdf5.lib")
	ENDIF ()
ELSEIF (APPLE)
	SET (INSTALLED_HDF5 ON)
ELSE ()
	SET (HDF5_VERSION "1.8.17")
	SET (HDF5_INCLUDE_DIRS "~/Public/hdf5-1.8.17/include")
        SET (HDF5_LIBRARIES "~/Public/hdf5-1.8.17/lib/libhdf5.so")
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

IF (WIN32)
FIND_PACKAGE (SZIP REQUIRED)
MESSAGE (" szip version major is " ${SZIP_VERSION_MAJOR})
MESSAGE (" szip version minor is " ${SZIP_VERSION_MINOR})
MESSAGE (" szip include is " ${SZIP_INCLUDE_DIR})
MESSAGE (" szip library is " ${SZIP_LIBRARY})
ENDIF ()

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
	set (Boost_INCLUDE_DIR ${APPLE_HOME}/Library/boost_1_55_0)
	set (Boost_LIBRARY_DIR ${APPLE_HOME}/Library/boost_1_55_0/stage/lib)
	include_directories (${APPLE_HOME}/Library/boost_1_55_0)
	set (Boost_USE_STATIC_LIBS ON)
	find_package(Boost 1.55 REQUIRED  COMPONENTS system filesystem date_time regex thread)
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

if (WIN32)
#add_definitions (-DOPENEXR_DLL)
set (OpenEXR_INCLUDE_DIR D:/usr/openexr21/include)
set (OpenEXR_Library "D:/usr/openexr21/lib/Half.lib"
"D:/usr/openexr21/lib/Iex-2_1.lib"
"D:/usr/openexr21/lib/IlmImf-2_1.lib"
"D:/usr/openexr21/lib/IlmThread-2_1.lib")
else ()
set (OpenEXR_INCLUDE_DIR /Users/jianzhang/Library/openexr21/include)
set (OpenEXR_Library /Users/jianzhang/Library/openexr21/lib/libHalf.dylib
 /Users/jianzhang/Library/openexr21/lib/libIex-2_1.dylib
 /Users/jianzhang/Library/openexr21/lib/libIlmImf-2_1.dylib
 /Users/jianzhang/Library/openexr21/lib/libIlmThread-2_1.dylib)
endif ()
include_directories (${OpenEXR_INCLUDE_DIR})
message (" openexr " ${OpenEXR_Library})


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

## https://cmake.org/Wiki/CMake:Bundles_And_Frameworks
set(CMAKE_MACOSX_BUNDLE ON)

add_subdirectory(rott)
add_subdirectory(naca4)
add_subdirectory(hermite)
add_subdirectory(pca)
add_subdirectory(hull)
add_subdirectory(cluster)
add_subdirectory(inst)
add_subdirectory(tetf)
add_subdirectory(abg)
add_subdirectory(tkm)
add_subdirectory(ttri)
add_subdirectory(exrs)
add_subdirectory(wbg)
