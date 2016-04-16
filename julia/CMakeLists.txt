cmake_minimum_required (VERSION 2.8)
project (julia)
SET (CMAKE_AUTOMOC ON)
SET (CMAKE_INCLUDE_CURRENT_DIR ON)
SET (CMAKE_BUILD_TYPE Release)
include_directories ("../shared" "../btree"
../ntree
../kdtree
../hesperis)

set (HEADER_FILES ../btree/HWorldGrid.h
../btree/HInnerGrid.h
../kdtree/HNTree.h
../kdtree/KdEngine.h
../kdtree/GridBuilder.h
../kdtree/VoxelGrid.h
../ntree/NTreeIO.h
../shared/HAsset.h
../btree/HAssetGrid.h
QuatJulia.h
JuliaTree.h
Parameter.h
JuliaWorld.h
)

set (SOURCE_FILES 
../shared/KdTreeNode.cpp
../shared/MinMaxBins.cpp 
../shared/SplitEvent.cpp
../shared/BaseBinSplit.cpp 
../shared/HElemAsset.cpp
../shared/CartesianGrid.cpp
../btree/GridClustering.cpp
../kdtree/TreeProperty.cpp
../kdtree/PrimBoundary.cpp
../kdtree/HNTree.cpp
../ntree/NTreeIO.cpp
../btree/HWorldGrid.cpp
Parameter.cpp 
QuatJulia.cpp
JuliaTree.cpp
JuliaWorld.cpp
main.cpp)

add_executable(julia ${HEADER_FILES} ${SOURCE_FILES})

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
target_link_libraries (julia ${HDF5_LIBRARIES})

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

target_link_libraries (julia ${ZLIB_LIBRARIES})

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

target_link_libraries (julia ${SZIP_LIBRARY})

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

FIND_PACKAGE (Boost REQUIRED COMPONENTS system filesystem date_time regex thread chrono)	
	
MESSAGE (" boost version is " ${Boost_VERSION})
MESSAGE (" boost include is " ${Boost_INCLUDE_DIR} )
MESSAGE (" boost system is " ${Boost_SYSTEM_LIBRARY})
MESSAGE (" boost date_time is " ${Boost_DATE_TIME_LIBRARY})
MESSAGE (" boost regex is " ${Boost_REGEX_LIBRARY})
MESSAGE (" boost filesystem is " ${Boost_FILESYSTEM_LIBRARY})
MESSAGE (" boost threaed is " ${Boost_THREAD_LIBRARY})
MESSAGE (" boost threaed is " ${Boost_CHRONO_LIBRARY})

include_directories (${Boost_INCLUDE_DIR})
	
target_link_libraries (julia ${Boost_SYSTEM_LIBRARY})
target_link_libraries (julia ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (julia ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (julia ${Boost_REGEX_LIBRARY})
target_link_libraries (julia ${Boost_THREAD_LIBRARY})
target_link_libraries (julia ${Boost_CHRONO_LIBRARY})

if (WIN32)
SET (Aphid_Math_Library D:/aphid/lib/aphidMath.lib)
SET (Aphid_Sdb_Library D:/aphid/lib/aphidSdb.lib)
SET (Aphid_H5_Library D:/aphid/lib/aphidH5.lib)
SET (Aphid_Geom_Library D:/aphid/lib/aphidGeom.lib)
SET (Aphid_Cu_Library D:/aphid/lib/aphidCu.lib)
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

target_link_libraries (julia ${Aphid_Math_Library})
target_link_libraries (julia ${Aphid_Sdb_Library})
target_link_libraries (julia ${Aphid_H5_Library})
target_link_libraries (julia ${Aphid_Geom_Library})

SET (VIEWERHEADER Parameter.h
viewerWindow.h
CubeRenderInterface.h
WorldRenderInterface.h
Manager.h
WorldManager.h
WorldRender.h
../kdtree/CudaNTree.h
viewerParameter.h
AssetRender.h
voxTest.h)

SET (VIEWERSOURCE viewerMain.cpp
renderthread.cpp
whitenoisewidget.cpp
CubeRender.cpp
../shared/KdTreeNode.cpp
../shared/MinMaxBins.cpp 
../shared/SplitEvent.cpp
../shared/BaseBinSplit.cpp
../shared/HElemAsset.cpp
../shared/CartesianGrid.cpp
../btree/GridClustering.cpp
../kdtree/TreeProperty.cpp
../kdtree/PrimBoundary.cpp
../kdtree/HNTree.cpp
../ntree/NTreeIO.cpp
../btree/HWorldGrid.cpp
viewerWindow.cpp
WorldRender.cpp
voxTest.cpp
AssetRender.cpp
viewerParameter.cpp)

SET (CUSOURCES 
../shared/cu/ImageBase.cu
cu/CubeRender.cu
cu/WorldRender.cu
)

find_package (CUDA QUIET REQUIRED)

message ("cuda found ? " ${CUDA_FOUND})
message ("cuda version " ${CUDA_VERSION})
message ("cuda sdk root dir " ${CUDA_SDK_ROOT_DIR})
message ("cuda librarie " ${CUDA_CUDA_LIBRARY})
message ("cuda art librarie " ${CUDA_CUDART_LIBRARY})
message ("cuda blas librarie " ${CUDA_cublas_LIBRARY})
message ("cuda sparse librarie " ${CUDA_cusparse_LIBRARY})

IF (WIN32)
	SET (CUDA_NVCC_FLAGS "--disable-warnings --ptxas-options=-v -arch sm_21")
ELSE ()
	SET (CUDA_NVCC_FLAGS "--ptxas-options=-v -arch sm_11")
ENDIF ()

CUDA_ADD_EXECUTABLE (viewJulia ${VIEWERHEADER} ${VIEWERSOURCE} ${CUSOURCES})

target_link_libraries(viewJulia ${Boost_LIBRARIES})
target_link_libraries (viewJulia ${Aphid_Math_Library})
target_link_libraries (viewJulia ${Aphid_Sdb_Library})
target_link_libraries (viewJulia ${Aphid_H5_Library})
target_link_libraries (viewJulia ${Aphid_Geom_Library})
target_link_libraries (viewJulia ${Aphid_Cu_Library})

if(WIN32)
## set (QT_QMAKE_EXECUTABLE "D:/usr/Qtx64/4.7.1/bin/qmake.exe")
set (QT_QMAKE_EXECUTABLE "D:/usr/qt4.8.6/bin/qmake.exe")
endif()

find_package (Qt4 REQUIRED)
message ("qt opengl " ${QT_QTOPENGL_LIBRARY})
SET (QT_USE_QTOPENGL ON)

target_link_libraries (viewJulia Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (viewJulia ${HDF5_LIBRARIES})
target_link_libraries (viewJulia ${ZLIB_LIBRARIES})
target_link_libraries (viewJulia ${SZIP_LIBRARY})
