## http://www.cmake.org/cmake/help/v2.8.12/cmake.html#module:FindQt4
## http://www.cmake.org/cmake/help/v3.0/manual/cmake-qt.7.html#introduction

cmake_minimum_required (VERSION 2.8)
project (foo)
set (CMAKE_AUTOMOC ON)
set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_BUILD_TYPE RELEASE)
include_directories (../shared
../bcc2d
../hesperis
../kdtree
../btree)

set (HEADER_FILES "../shared/Base3DView.h" 
"window.h"
"glwidget.h"
"../shared/Morton3D.h"
../shared/hilbertCurve.h
../shared/triangle_math.h
../shared/tetrahedron_math.h
../shared/AGraph.h
../shared/ADistanceField.h
../shared/BDistanceFunction.h
../btree/AdaptiveGrid3.h
../btree/GridTables.h
triangulation.h
tetrahedron_graph.h
bipyramid_flip.h
diamond_flip.h
tetrahedralization.h
SuperformulaControl.h
Parameter.h
Scene.h
Hilbert2D.h
Hilbert3D.h
Delaunay2D.h
ClosestSampleTest.h
Convexity.h
BlueYellowCyanRefine.h
BccCell.h
BccTetraGrid.h
Bcc3dTest.h
TetrahedralMesher.h
SuperformulaTest.h
PoissonSequence.h
SuperformulaPoisson.h
BccTetrahedralize.h
GridMaker.h
TetraField.h
DistanceFieldTest.h
RedBlueRefine.h
RedBlueTest.h
RedBlueControl.h
FieldTetrahedralizeTest.h
BccCell3.h
AdaptiveBccGrid3.h
AdaptiveBccMesher.h
AdaptiveBccField.h
FieldTriangulation.h
AdaptiveGridTest.h
Delaunay3D.h)

set (SOURCE_FILES ../shared/Base3DView.cpp
../shared/CartesianGrid.cpp
../shared/QDoubleEditSlider.cpp
SuperformulaControl.cpp
Parameter.cpp
Scene.cpp
Hilbert2D.cpp
Delaunay2D.cpp
Delaunay3D.cpp
Hilbert3D.cpp
ClosestSampleTest.cpp
Convexity.cpp
BlueYellowCyanRefine.cpp
BccCell.cpp
BccTetraGrid.cpp
TetrahedralMesher.cpp
Bcc3dTest.cpp
SuperformulaTest.cpp
SuperformulaPoisson.cpp
BccTetrahedralize.cpp
GridMaker.cpp
TetraField.cpp
DistanceFieldTest.cpp
RedBlueRefine.cpp
RedBlueTest.cpp
RedBlueControl.cpp
FieldTetrahedralizeTest.cpp
BccCell3.cpp
AdaptiveBccGrid3.cpp
AdaptiveBccMesher.cpp
AdaptiveBccField.cpp
FieldTriangulation.cpp
AdaptiveGridTest.cpp
glwidget.cpp
window.cpp
main.cpp)

if (WIN32)
set (SOURCE_FILES ${SOURCE_FILES} "../shared/gExtension.cpp")
endif ()

add_executable (foo ${HEADER_FILES} ${SOURCE_FILES})

find_package (OpenGL REQUIRED)
target_link_libraries (foo ${OPENGL_LIBRARIES})

if (WIN32)
set (Aphid_Library D:/aphid/lib/aphidMath.lib
D:/aphid/lib/aphidGeom.lib
D:/aphid/lib/aphidSdb.lib
D:/aphid/lib/aphidGraph.lib
D:/aphid/lib/aphidUI.lib)
else ()
set (Aphid_Library /users/jianzhang/aphid/lib/libaphidMath.a
/users/jianzhang/aphid/lib/libaphidGeom.a
/users/jianzhang/aphid/lib/libaphidSdb.a
/users/jianzhang/aphid/lib/libaphidGraph.a
/users/jianzhang/aphid/lib/libaphidUI.a)
endif ()
target_link_libraries (foo ${Aphid_Library})

if (WIN32)
set (OpenEXR_Library "D:/usr/openEXR/lib/Half.lib"
"D:/usr/openEXR/lib/Iex.lib"
"D:/usr/openEXR/lib/IlmImf.lib"
"D:/usr/openEXR/lib/IlmThread.lib")  
else()
set (OpenEXR_Library "/usr/local/lib/libHalf.a"
"/usr/local/lib/libIex.a"
"/usr/local/lib/libIlmImf.a"
"/usr/local/lib/libIlmThread.a")
endif()

target_link_libraries (foo ${OpenEXR_Library})

find_package (zlib REQUIRED)
target_link_libraries (foo ${ZLIB_LIBRARIES})

if(WIN32)
set (QT_QMAKE_EXECUTABLE "D:/usr/qt4.8.6/bin/qmake.exe")
endif()

find_package (Qt4 REQUIRED)
message ("qt opengl " ${QT_QTOPENGL_LIBRARY})
set (QT_USE_QTOPENGL ON)
target_link_libraries (foo Qt4::QtGui Qt4::QtOpenGL)

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
	find_package(Boost 1.55 COMPONENTS system filesystem date_time regex thread chrono)
	include_directories ("/usr/local/include")

else()
	set (Boost_INCLUDE_DIR "/home/OF3D/zhangjian/Documents/boost_1_51_0")
	set (Boost_USE_STATIC_LIBS ON)
	set (Boost_USE_MULTITHREADED ON)
	include_directories ("/home/OF3D/zhangjian/Documents/boost_1_51_0")
	find_package(Boost 1.51 COMPONENTS system filesystem date_time regex thread chrono)	
endif()
	
endif()

message ("boost system is" ${Boost_SYSTEM_LIBRARY})
message ("boost date_time is" ${Boost_DATE_TIME_LIBRARY})
message ("boost regex is" ${Boost_REGEX_LIBRARY})
message ("boost filesystem is" ${Boost_FILESYSTEM_LIBRARY})
message ("boost thread is" ${Boost_THREAD_LIBRARY})

target_link_libraries (foo ${Boost_SYSTEM_LIBRARY})
target_link_libraries (foo ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (foo ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (foo ${Boost_REGEX_LIBRARY})
target_link_libraries (foo ${Boost_THREAD_LIBRARY})
target_link_libraries (foo ${Boost_CHRONO_LIBRARY})

set (AttributeNoninline "__attribute__ ((noinline))")
add_definitions (-DBOOST_NOINLINE=${AttributeNoninline})

message (${Boost_LIBRARIES})
target_link_libraries(foo ${Boost_LIBRARIES})

SET (EXECUTABLE_OUTPUT_PATH ../bin)
