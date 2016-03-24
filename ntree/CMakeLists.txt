## http://www.cmake.org/cmake/help/v2.8.12/cmake.html#module:FindQt4
## http://www.cmake.org/cmake/help/v3.0/manual/cmake-qt.7.html#introduction

cmake_minimum_required (VERSION 2.8)
project (testntree)
set (CMAKE_AUTOMOC ON)
set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_BUILD_TYPE RELEASE)
include_directories (../shared
../bcc2d
../hesperis
../kdtree
../btree
../lapl)

set (HEADER_FILES "../shared/Base3DView.h" 
"window.h"
"glwidget.h"
"../shared/Morton3D.h"
../shared/ConvexShape.h
../shared/HElemAsset.h
../kdtree/KdBuilder.h
../kdtree/KdEngine.h
../kdtree/KdRope.h
../kdtree/KdNTree.h
../kdtree/KdNNode.h
../kdtree/KdSah.h
../kdtree/TreeProperty.h
../kdtree/HNTree.h
../kdtree/VoxelGrid.h
NTreeIO.h
NTreeDrawer.h)

set (SOURCE_FILES ../shared/Base3DView.cpp
../shared/KdTreeNode.cpp
../shared/MinMaxBins.cpp 
../shared/SplitEvent.cpp
../shared/BaseBinSplit.cpp 
../shared/GeoDrawer.cpp
../shared/ViewFrame.cpp
../shared/DrawBox.cpp
../shared/CartesianGrid.cpp
../shared/HElemAsset.cpp
../kdtree/TreeProperty.cpp
../kdtree/PrimBoundary.cpp
../btree/HWorldGrid.cpp
../kdtree/HNTree.cpp
NTreeIO.cpp
glwidget.cpp
window.cpp
main.cpp)

if (WIN32)
set (SOURCE_FILES ${SOURCE_FILES} "../shared/gExtension.cpp")
endif ()

add_executable (testntree ${HEADER_FILES} ${SOURCE_FILES})

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
target_link_libraries (testntree ${HDF5_LIBRARIES})

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

target_link_libraries (testntree ${ZLIB_LIBRARIES})

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

target_link_libraries (testntree ${SZIP_LIBRARY})

find_package (OpenGL REQUIRED)
target_link_libraries (testntree ${OPENGL_LIBRARIES})

if (WIN32)
set (Aphid_Library D:/aphid/lib/aphidMath.lib
D:/aphid/lib/aphidGeom.lib
D:/aphid/lib/aphidH5.lib
D:/aphid/lib/aphidSdb.lib
D:/aphid/lib/aphidUI.lib)
else ()
set (Aphid_Library /users/jianzhang/aphid/lib/libaphidMath.a
/users/jianzhang/aphid/lib/libaphidGeom.a
/users/jianzhang/aphid/lib/libaphidUI.a
/users/jianzhang/aphid/lib/libaphidH5.a
/users/jianzhang/aphid/lib/libaphidSdb.a)
endif ()

target_link_libraries (testntree ${Aphid_Library})

if(WIN32)
set (QT_QMAKE_EXECUTABLE "D:/usr/qt4.8.6/bin/qmake.exe")
endif()

find_package (Qt4 REQUIRED)
message ("qt opengl " ${QT_QTOPENGL_LIBRARY})
set (QT_USE_QTOPENGL ON)
target_link_libraries (testntree Qt4::QtGui Qt4::QtOpenGL)

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

target_link_libraries (testntree ${Boost_SYSTEM_LIBRARY})
target_link_libraries (testntree ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (testntree ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (testntree ${Boost_REGEX_LIBRARY})
target_link_libraries (testntree ${Boost_THREAD_LIBRARY})
target_link_libraries (testntree ${Boost_CHRONO_LIBRARY})

set (AttributeNoninline "__attribute__ ((noinline))")
add_definitions (-DBOOST_NOINLINE=${AttributeNoninline})

set (SHOTRI_HEADER_FILES "../shared/Base3DView.h" 
triWindow.h
"../shared/Morton3D.h"
../shared/ConvexShape.h
../shared/HElemAsset.h
../kdtree/KdBuilder.h
../kdtree/KdEngine.h
../kdtree/KdRope.h
../kdtree/KdNTree.h
../kdtree/KdNNode.h
../kdtree/KdSah.h
../kdtree/TreeProperty.h
../kdtree/HNTree.h
../kdtree/VoxelGrid.h
NTreeIO.h
NTreeDrawer.h
Container.h
triWidget.h)

set (SHOTRI_SOURCE_FILES ../shared/Base3DView.cpp
../shared/KdTreeNode.cpp
../shared/MinMaxBins.cpp 
../shared/SplitEvent.cpp
../shared/BaseBinSplit.cpp 
../shared/GeoDrawer.cpp
../shared/ViewFrame.cpp
../shared/DrawBox.cpp
../shared/CartesianGrid.cpp
../shared/HElemAsset.cpp
../kdtree/TreeProperty.cpp
../kdtree/PrimBoundary.cpp
../btree/HWorldGrid.cpp
../kdtree/HNTree.cpp
NTreeIO.cpp
triWidget.cpp
triWindow.cpp
triMain.cpp)

if (WIN32)
set (SHOTRI_SOURCE_FILES ${SOURCE_FILES} "../shared/gExtension.cpp")
endif ()

add_executable (shotriangle ${SHOTRI_HEADER_FILES} ${SHOTRI_SOURCE_FILES})
target_link_libraries (shotriangle ${HDF5_LIBRARIES})
target_link_libraries (shotriangle ${ZLIB_LIBRARIES})
target_link_libraries (shotriangle ${SZIP_LIBRARY})
target_link_libraries (shotriangle ${Aphid_Library})
target_link_libraries (shotriangle ${Boost_SYSTEM_LIBRARY})
target_link_libraries (shotriangle ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (shotriangle ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (shotriangle ${Boost_REGEX_LIBRARY})
target_link_libraries (shotriangle ${Boost_THREAD_LIBRARY})
target_link_libraries (shotriangle ${Boost_CHRONO_LIBRARY})
target_link_libraries (shotriangle Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (shotriangle ${OPENGL_LIBRARIES})



