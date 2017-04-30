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
../btree
../ntree)

set (HEADER_FILES ../shared/Base3DView.h 
../shared/Morton3D.h
../shared/ConvexShape.h
../shared/HElemAsset.h
../shared/VoxelEngine.h
../kdtree/KdBuilder.h
../kdtree/KdEngine.h
../kdtree/KdRope.h
../kdtree/KdNTree.h
../kdtree/KdNNode.h
../kdtree/KdSah.h
../kdtree/HNTree.h
../kdtree/VoxelGrid.h
window.h
glwidget.h
../shared/hilbertCurve.h
../shared/triangle_math.h
../shared/tetrahedron_math.h
../shared/AGraph.h
../shared/ADistanceField.h
../shared/BDistanceFunction.h
../btree/AdaptiveGrid3.h
../btree/GridTables.h
../ntree/Container.h
../ntree/NTreeIO.h
../ntree/NTreeDrawer.h
../ntree/NTreeDomain.h
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
KDistanceTest.h
VDGridTest.h
Noise3Test.h
LegendreTest.h
Legendre2DTest.h
Legendre3DTest.h
IntersectTest.h
Delaunay3D.h)

set (SOURCE_FILES ../shared/Base3DView.cpp
../shared/QDoubleEditSlider.cpp
../shared/HElemAsset.cpp
../btree/HWorldGrid.cpp
../ntree/NTreeIO.cpp
../kdtree/HNTree.cpp
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
KDistanceTest.cpp
VDGridTest.cpp
Noise3Test.cpp
LegendreTest.cpp
Legendre2DTest.cpp
Legendre3DTest.cpp
IntersectTest.cpp
glwidget.cpp
window.cpp
main.cpp)

if (WIN32)
set (SOURCE_FILES ${SOURCE_FILES} "../shared/gExtension.cpp")
endif ()

add_executable (foo ${HEADER_FILES} ${SOURCE_FILES})

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
target_link_libraries (foo ${HDF5_LIBRARIES})

if(WIN32)
##SET (ZLIB_DIR "C:/Program Files/ZLIB/cmake/ZLIB")
##SET (ZLIB_INCLUDE_DIR "C:/Program Files/ZLIB/include")
##SET (ZLIB_LIBRARY "C:/Program Files/ZLIB/lib/libzlib.lib")
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
target_link_libraries (foo ${ZLIB_LIBRARIES})
ENDIF ()

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

target_link_libraries (foo ${SZIP_LIBRARY})
ENDIF ()

find_package (OpenGL REQUIRED)
target_link_libraries (foo ${OPENGL_LIBRARIES})

if (WIN32)
SET (APHID_DIR "D:/aphid")
SET (APHID_MATH_LIBRARY ${APHID_DIR}/lib/aphidMath.lib)
    SET (APHID_GEOM_LIBRARY ${APHID_DIR}/lib/aphidGeom.lib)
    SET (APHID_H5_LIBRARY ${APHID_DIR}/lib/aphidH5.lib)
    SET (APHID_UI_LIBRARY ${APHID_DIR}/lib/aphidUI.lib)
    SET (APHID_KD_LIBRARY ${APHID_DIR}/lib/aphidKd.lib)
    SET (APHID_SDB_LIBRARY ${APHID_DIR}/lib/aphidSdb.lib)
    SET (APHID_GRAPH_LIBRARY ${APHID_DIR}/lib/aphidGraph.lib)
elseif (APPLE)
set (Aphid_Library /users/jianzhang/aphid/lib/libaphidMath.a
/users/jianzhang/aphid/lib/libaphidGeom.a
/users/jianzhang/aphid/lib/libaphidH5.a
/users/jianzhang/aphid/lib/libaphidSdb.a
/users/jianzhang/aphid/lib/libaphidGraph.a
/users/jianzhang/aphid/lib/libaphidKd.a
/users/jianzhang/aphid/lib/libaphidUI.a)
ELSE ()
    SET (APHID_DIR "~/aphid")
    SET (APHID_MATH_LIBRARY ${APHID_DIR}/lib/libaphidMath.so)
    SET (APHID_GEOM_LIBRARY ${APHID_DIR}/lib/libaphidGeom.so)
    SET (APHID_H5_LIBRARY ${APHID_DIR}/lib/libaphidH5.so)
    SET (APHID_UI_LIBRARY ${APHID_DIR}/lib/libaphidUI.so)
    SET (APHID_KD_LIBRARY ${APHID_DIR}/lib/libaphidKd.so)
    SET (APHID_SDB_LIBRARY ${APHID_DIR}/lib/libaphidSdb.so)
    SET (APHID_GRAPH_LIBRARY ${APHID_DIR}/lib/libaphidGraph.so)
endif ()

target_link_libraries (foo ${APHID_MATH_LIBRARY})
target_link_libraries (foo ${APHID_GEOM_LIBRARY})
target_link_libraries (foo ${APHID_UI_LIBRARY})
target_link_libraries (foo ${APHID_KD_LIBRARY})
target_link_libraries (foo ${APHID_SDB_LIBRARY})
target_link_libraries (foo ${APHID_GRAPH_LIBRARY})
target_link_libraries (foo ${APHID_H5_LIBRARY})

if (WIN32)
add_definitions (-DOPENEXR_DLL)
set (OpenEXR_Library "D:/usr/openEXR/lib/Half.lib"
"D:/usr/openEXR/lib/Iex.lib"
"D:/usr/openEXR/lib/IlmImf.lib"
"D:/usr/openEXR/lib/IlmThread.lib")  
elseif (APPLE)
set (OpenEXR_INCLUDE_DIR /Users/jianzhang/Library/openexr21/include)
set (OpenEXR_Library /Users/jianzhang/Library/openexr21/lib/libHalf.dylib
 /Users/jianzhang/Library/openexr21/lib/libIex-2_1.dylib
 /Users/jianzhang/Library/openexr21/lib/libIlmImf-2_1.dylib
 /Users/jianzhang/Library/openexr21/lib/libIlmThread-2_1.dylib)
else()
SET (ILMBASE_PACKAGE_PREFIX /usr/local)

set (OpenEXR_INCLUDE_DIR ${ILMBASE_PACKAGE_PREFIX}/include)
set (OpenEXR_Library ${ILMBASE_PACKAGE_PREFIX}/lib/libHalf.so
    ${ILMBASE_PACKAGE_PREFIX}/lib/libIex-2_2.so
    ${ILMBASE_PACKAGE_PREFIX}/lib/libImath-2_2.so
    ${ILMBASE_PACKAGE_PREFIX}/lib/libIlmImf-2_2.so
    ${ILMBASE_PACKAGE_PREFIX}/lib/libIlmThread-2_2.so)
endif ()

target_link_libraries (foo ${OpenEXR_Library})

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

message ("boost system is    " ${Boost_SYSTEM_LIBRARY})
message ("boost date_time is " ${Boost_DATE_TIME_LIBRARY})
message ("boost regex is     " ${Boost_REGEX_LIBRARY})
message ("boost filesystem is" ${Boost_FILESYSTEM_LIBRARY})
message ("boost thread is    " ${Boost_THREAD_LIBRARY})

target_link_libraries (foo ${Boost_SYSTEM_LIBRARY})
target_link_libraries (foo ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (foo ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (foo ${Boost_REGEX_LIBRARY})
target_link_libraries (foo ${Boost_THREAD_LIBRARY})
target_link_libraries (foo ${Boost_CHRONO_LIBRARY})

set (AttributeNoninline "__attribute__ ((noinline))")
add_definitions (-DBOOST_NOINLINE=${AttributeNoninline})

set (VDF_HEADER_FILES ../shared/Base3DView.h 
../shared/Morton3D.h
../shared/ConvexShape.h
../shared/HElemAsset.h
../shared/VoxelEngine.h
../kdtree/KdBuilder.h
../kdtree/KdEngine.h
../kdtree/KdRope.h
../kdtree/KdNTree.h
../kdtree/KdNNode.h
../kdtree/KdSah.h
../kdtree/HNTree.h
../kdtree/VoxelGrid.h
../shared/hilbertCurve.h
../shared/triangle_math.h
../shared/tetrahedron_math.h
../shared/AGraph.h
../shared/ADistanceField.h
../shared/BDistanceFunction.h
../btree/AdaptiveGrid3.h
../btree/GridTables.h
../ntree/Container.h
../ntree/NTreeIO.h
../ntree/NTreeDrawer.h
../ntree/NTreeDomain.h
triangulation.h
tetrahedron_graph.h
tetrahedralization.h
vdfParameter.h
Scene.h
RedBlueRefine.h
BccCell3.h
AdaptiveBccGrid3.h
AdaptiveBccMesher.h
AdaptiveBccField.h
FieldTriangulation.h
vdfScene.h
vdfWindow.h
vdfWidget.h)

set (VDF_SOURCE_FILES ../shared/Base3DView.cpp
../shared/HElemAsset.cpp
../btree/HWorldGrid.cpp
../ntree/NTreeIO.cpp
../kdtree/HNTree.cpp
vdfParameter.cpp
Scene.cpp
RedBlueRefine.cpp
BccCell3.cpp
AdaptiveBccGrid3.cpp
AdaptiveBccMesher.cpp
AdaptiveBccField.cpp
FieldTriangulation.cpp
vdfScene.cpp
vdfWidget.cpp
vdfWindow.cpp
vdfMain.cpp)

if (WIN32)
set (VDF_SOURCE_FILES ${VDF_SOURCE_FILES} "../shared/gExtension.cpp")
endif ()

add_executable (vdf ${VDF_HEADER_FILES} ${VDF_SOURCE_FILES})

target_link_libraries (vdf ${HDF5_LIBRARIES})
target_link_libraries (vdf ${ZLIB_LIBRARIES})
target_link_libraries (vdf Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (vdf ${SZIP_LIBRARY})
target_link_libraries (vdf ${OPENGL_LIBRARIES})
target_link_libraries (vdf ${APHID_MATH_LIBRARY})
target_link_libraries (vdf ${APHID_GEOM_LIBRARY})
target_link_libraries (vdf ${APHID_UI_LIBRARY})
target_link_libraries (vdf ${APHID_KD_LIBRARY})
target_link_libraries (vdf ${APHID_SDB_LIBRARY})
target_link_libraries (vdf ${APHID_GRAPH_LIBRARY})
target_link_libraries (vdf ${APHID_H5_LIBRARY})
target_link_libraries (vdf ${Boost_SYSTEM_LIBRARY})
target_link_libraries (vdf ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (vdf ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (vdf ${Boost_REGEX_LIBRARY})
target_link_libraries (vdf ${Boost_THREAD_LIBRARY})
target_link_libraries (vdf ${Boost_CHRONO_LIBRARY})

set (HEG_SOURCE_FILES ../shared/Base3DView.cpp
hegWidget.cpp
hegWindow.cpp
hegMain.cpp)

add_executable (hexasplit ${HEG_SOURCE_FILES})

target_link_libraries (hexasplit Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (hexasplit ${OPENGL_LIBRARIES})
target_link_libraries (hexasplit ${APHID_MATH_LIBRARY})
target_link_libraries (hexasplit ${APHID_GEOM_LIBRARY})
target_link_libraries (hexasplit ${APHID_UI_LIBRARY})
target_link_libraries (hexasplit ${APHID_KD_LIBRARY})
target_link_libraries (hexasplit ${APHID_SDB_LIBRARY})
target_link_libraries (hexasplit ${APHID_GRAPH_LIBRARY})
target_link_libraries (hexasplit ${Boost_SYSTEM_LIBRARY})
target_link_libraries (hexasplit ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (hexasplit ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (hexasplit ${Boost_REGEX_LIBRARY})
target_link_libraries (hexasplit ${Boost_THREAD_LIBRARY})
target_link_libraries (hexasplit ${Boost_CHRONO_LIBRARY})

SET (EXECUTABLE_OUTPUT_PATH ../bin)
