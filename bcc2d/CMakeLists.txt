## http://www.cmake.org/cmake/help/v2.8.12/cmake.html#module:FindQt4
## http://www.cmake.org/cmake/help/v3.0/manual/cmake-qt.7.html#introduction

cmake_minimum_required (VERSION 2.8)
project (bcc)
set (CMAKE_AUTOMOC ON)
set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_BUILD_TYPE RELEASE)

include_directories ("../shared" 
"../cudabvh" 
"../radixsort" 
"../cudanarrowphase"
"../cudadyn"
"../cudafem"
"../hesperis"
"../kdtree"
"../catmullclark"
"../btree")

set (HEADER_FILES "../shared/Base3DView.h" 
"../shared/Morton3D.h"
"../hesperis/HesperisFile.h"
"window.h"
"glwidget.h"
"bcc_common.h"
"BccLattice.h"
"BccGrid.h"
"BccMesh.h"
"BccWorld.h"
"BccOctahedron.h"
"CurveSampler.h"
"SampleGroup.h"
"CurveReduction.h"
"FitBccMeshBuilder.h"
GenTetControl.h
BccInterface.h)

set (SOURCE_FILES "../shared/Base3DView.cpp"
"../hesperis/HesperisFile.cpp"
"../kdtree/RandomCurve.cpp"
../shared/QIntEditSlider.cpp
../shared/QDoubleEditSlider.cpp
../shared/QSplineEdit.cpp
../shared/QAngleEdit.cpp
../shared/QPolarCoordinateEdit.cpp
"main.cpp"
"glWidget.cpp"
"window.cpp"
"BccWorld.cpp"
"BccOctahedron.cpp"
"CurveSampler.cpp"
"SampleGroup.cpp"
"CurveReduction.cpp"
"FitBccMeshBuilder.cpp"
../hesperis/HesperisFile.cpp
../hesperis/HesperisInterface.cpp
BlockBccMeshBuilder.cpp
GenTetControl.cpp
BccInterface.cpp)

if (WIN32)
set (SOURCE_FILES ${SOURCE_FILES} "../shared/gExtension.cpp")
endif ()

add_executable (${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

## find_package(HDF5 REQUIRED)
if (WIN32)
set (HDF5_INCLUDE_DIR "D:/usr/hdf5/include")
set (HDF5_LIBRARIES "D:/usr/hdf5/lib/hdf5.lib")
else ()
set (HDF5_INCLUDE_DIR "/users/jianzhang/Library/hdf5/include")
set (HDF5_LIBRARIES "/users/jianzhang/Library/hdf5/lib/libhdf5.dylib")
endif ()

if (WIN32)
set (SZIP_INCLUDE_DIR "D:/usr/szip-2.1/src")
set (SZIP_LIBRARIES "D:/usr/szip-2.1/szip.lib")
include_directories (${SZIP_INCLUDE_DIR})
target_link_libraries (${PROJECT_NAME} ${SZIP_LIBRARIES})
endif ()

include_directories (${HDF5_INCLUDE_DIR})
target_link_libraries (${PROJECT_NAME} ${HDF5_LIBRARIES})

find_package (OpenGL REQUIRED)
target_link_libraries (${PROJECT_NAME} ${OPENGL_LIBRARIES})


if (WIN32)
set (Aphid_Library "D:/aphid/lib/aphid.lib")
else ()
set (Aphid_Library "/users/jianzhang/aphid/lib/libaphid.a")
endif ()

target_link_libraries (${PROJECT_NAME} ${Aphid_Library})

if (WIN32)
set (OpenEXR_Library "D:/usr/openEXR/lib/Half.lib"
"D:/usr/openEXR/lib/Iex.lib"
"D:/usr/openEXR/lib/IlmImf.lib"
"D:/usr/openEXR/lib/IlmThread.lib")
else ()
set (OpenEXR_Library "/usr/local/lib/libHalf.a"
"/usr/local/lib/libIex.a"
"/usr/local/lib/libIlmImf.a"
"/usr/local/lib/libIlmThread.a")
endif ()
target_link_libraries (${PROJECT_NAME} ${OpenEXR_Library})

if(WIN32)
set (ZLIB_ROOT "D:/usr/zlib-1.2.5")
endif()

find_package (zlib REQUIRED)
target_link_libraries (${PROJECT_NAME} ${ZLIB_LIBRARIES})

if(WIN32)
## set (QT_QMAKE_EXECUTABLE "D:/usr/Qtx64/4.7.1/bin/qmake.exe")
set (QT_QMAKE_EXECUTABLE "D:/usr/qt4.8.6/bin/qmake.exe")
endif()

find_package (Qt4 REQUIRED)
message ("qt include dir " ${QT_INCLUDE_DIR})
message ("qt opengl " ${QT_QTOPENGL_LIBRARY})
set (QT_MT_REQUIRED ON)
set (QT_USE_QTOPENGL ON)

include_directories (${QT_INCLUDE_DIR})
include (${QT_USE_FILE})

## target_link_libraries (testcudafem Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES})

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
	include_directories ("/users/jianzhang/Library/boost_1_55_0")

else()
	set (Boost_INCLUDE_DIR "/home/OF3D/zhangjian/Documents/boost_1_51_0")
	set (Boost_USE_STATIC_LIBS ON)
	set (Boost_USE_MULTITHREADED ON)
	include_directories ("/home/OF3D/zhangjian/Documents/boost_1_51_0")
	find_package(Boost 1.51 COMPONENTS system filesystem date_time regex thread)	
endif()
	
endif()

message ("boost system is" ${Boost_SYSTEM_LIBRARY})
message ("boost date_time is" ${Boost_DATE_TIME_LIBRARY})
message ("boost regex is" ${Boost_REGEX_LIBRARY})
message ("boost filesystem is" ${Boost_FILESYSTEM_LIBRARY})
message ("boost thread is" ${Boost_THREAD_LIBRARY})

set (AttributeNoninline "__attribute__ ((noinline))")
add_definitions (-DBOOST_NOINLINE=${AttributeNoninline})

message (${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
