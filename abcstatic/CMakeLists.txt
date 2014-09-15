
cmake_minimum_required (VERSION 2.8)
project (alembic)

set (CMAKE_BUILD_TYPE Release)

add_definitions (-DLINUX -DUNIX -D_BOOL -DFUNCPROTO -D_GNU_SOURCE -DREQUIRE_IOSTREAM -DBits64_)

message (CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR})
ADD_DEFINITIONS(-fPIC -m64 -O3 -pthread)

#message (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
 
# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")

set (HDF5_INCLUDE_DIR /home/OF3D/zhangjian/Documents/hdf5-1.8.9/hdf5/include)
include_directories (${HDF5_INCLUDE_DIR})
message (" hdf5 inc is" ${HDF5_INCLUDE_DIR})

set (ABC_SRC /home/OF3D/zhangjian/Documents/alembic/lib/Alembic)
include_directories (/home/OF3D/zhangjian/Documents/alembic/lib)

set (BOOST_ROOT /home/OF3D/zhangjian/Documents/boost_1_51_0)
include_directories (${BOOST_ROOT})

set (SOURCES ${ABC_SRC}/Abc/ArchiveInfo.cpp 
                ${ABC_SRC}/Abc/ErrorHandler.cpp 
                ${ABC_SRC}/Abc/IArchive.cpp 
                ${ABC_SRC}/Abc/IArrayProperty.cpp 
                ${ABC_SRC}/Abc/ICompoundProperty.cpp 
                ${ABC_SRC}/Abc/IObject.cpp 
                ${ABC_SRC}/Abc/ISampleSelector.cpp 
                ${ABC_SRC}/Abc/IScalarProperty.cpp 
                ${ABC_SRC}/Abc/OArchive.cpp 
                ${ABC_SRC}/Abc/OArrayProperty.cpp 
                ${ABC_SRC}/Abc/OCompoundProperty.cpp 
                ${ABC_SRC}/Abc/OObject.cpp 
                ${ABC_SRC}/Abc/OScalarProperty.cpp 
                ${ABC_SRC}/Abc/Reference.cpp 
                ${ABC_SRC}/Abc/SourceName.cpp 
                ${ABC_SRC}/AbcCollection/ICollections.cpp 
                ${ABC_SRC}/AbcCollection/OCollections.cpp 
                ${ABC_SRC}/AbcCoreAbstract/Foundation.cpp 
                ${ABC_SRC}/AbcCoreAbstract/TimeSampling.cpp 
                ${ABC_SRC}/AbcCoreAbstract/TimeSamplingType.cpp 
                ${ABC_SRC}/AbcCoreAbstract/ArraySample.cpp 
                ${ABC_SRC}/AbcCoreAbstract/ReadArraySampleCache.cpp 
                ${ABC_SRC}/AbcCoreAbstract/ScalarSample.cpp 
                ${ABC_SRC}/AbcCoreAbstract/BasePropertyWriter.cpp 
                ${ABC_SRC}/AbcCoreAbstract/ScalarPropertyWriter.cpp 
                ${ABC_SRC}/AbcCoreAbstract/ArrayPropertyWriter.cpp 
                ${ABC_SRC}/AbcCoreAbstract/CompoundPropertyWriter.cpp 
                ${ABC_SRC}/AbcCoreAbstract/ObjectWriter.cpp  
                ${ABC_SRC}/AbcCoreAbstract/ArchiveWriter.cpp  
                ${ABC_SRC}/AbcCoreAbstract/BasePropertyReader.cpp 
                ${ABC_SRC}/AbcCoreAbstract/ScalarPropertyReader.cpp 
                ${ABC_SRC}/AbcCoreAbstract/ArrayPropertyReader.cpp 
                ${ABC_SRC}/AbcCoreAbstract/CompoundPropertyReader.cpp 
                ${ABC_SRC}/AbcCoreAbstract/ObjectReader.cpp  
                ${ABC_SRC}/AbcCoreAbstract/ArchiveReader.cpp  
                ${ABC_SRC}/AbcCoreFactory/IFactory.cpp 
                ${ABC_SRC}/AbcCoreHDF5/AprImpl.cpp 
                ${ABC_SRC}/AbcCoreHDF5/ApwImpl.cpp 
                ${ABC_SRC}/AbcCoreHDF5/ArImpl.cpp 
                ${ABC_SRC}/AbcCoreHDF5/AwImpl.cpp 
                ${ABC_SRC}/AbcCoreHDF5/CacheImpl.cpp 
                ${ABC_SRC}/AbcCoreHDF5/CprData.cpp 
                ${ABC_SRC}/AbcCoreHDF5/CprImpl.cpp 
                ${ABC_SRC}/AbcCoreHDF5/CpwData.cpp 
                ${ABC_SRC}/AbcCoreHDF5/CpwImpl.cpp 
                ${ABC_SRC}/AbcCoreHDF5/DataTypeRegistry.cpp 
                ${ABC_SRC}/AbcCoreHDF5/HDF5Hierarchy.cpp 
                ${ABC_SRC}/AbcCoreHDF5/HDF5HierarchyReader.cpp 
                ${ABC_SRC}/AbcCoreHDF5/HDF5HierarchyWriter.cpp 
                ${ABC_SRC}/AbcCoreHDF5/HDF5Util.cpp 
                ${ABC_SRC}/AbcCoreHDF5/OrData.cpp 
                ${ABC_SRC}/AbcCoreHDF5/OrImpl.cpp 
                ${ABC_SRC}/AbcCoreHDF5/OwData.cpp 
                ${ABC_SRC}/AbcCoreHDF5/OwImpl.cpp 
                ${ABC_SRC}/AbcCoreHDF5/ReadUtil.cpp 
                ${ABC_SRC}/AbcCoreHDF5/ReadWrite.cpp 
                ${ABC_SRC}/AbcCoreHDF5/SprImpl.cpp 
                ${ABC_SRC}/AbcCoreHDF5/SpwImpl.cpp 
                ${ABC_SRC}/AbcCoreHDF5/StringReadUtil.cpp 
                ${ABC_SRC}/AbcCoreHDF5/StringWriteUtil.cpp 
                ${ABC_SRC}/AbcCoreHDF5/WriteUtil.cpp 
                ${ABC_SRC}/Util/Murmur3.cpp 
     		${ABC_SRC}/Util/Naming.cpp 
		${ABC_SRC}/Util/TokenMap.cpp 
     		${ABC_SRC}/Util/SpookyV2.cpp 
                ${ABC_SRC}/AbcGeom/ArchiveBounds.cpp 
                ${ABC_SRC}/AbcGeom/GeometryScope.cpp 
                ${ABC_SRC}/AbcGeom/FilmBackXformOp.cpp 
                ${ABC_SRC}/AbcGeom/CameraSample.cpp 
                ${ABC_SRC}/AbcGeom/ICamera.cpp 
                ${ABC_SRC}/AbcGeom/OCamera.cpp 
                ${ABC_SRC}/AbcGeom/Basis.cpp 
                ${ABC_SRC}/AbcGeom/ICurves.cpp 
                ${ABC_SRC}/AbcGeom/OCurves.cpp 
                ${ABC_SRC}/AbcGeom/OFaceSet.cpp 
                ${ABC_SRC}/AbcGeom/IFaceSet.cpp 
                ${ABC_SRC}/AbcGeom/OLight.cpp 
                ${ABC_SRC}/AbcGeom/ILight.cpp 
                ${ABC_SRC}/AbcGeom/ONuPatch.cpp 
                ${ABC_SRC}/AbcGeom/INuPatch.cpp 
                ${ABC_SRC}/AbcGeom/OPoints.cpp 
                ${ABC_SRC}/AbcGeom/IPoints.cpp 
                ${ABC_SRC}/AbcGeom/OPolyMesh.cpp 
                ${ABC_SRC}/AbcGeom/IPolyMesh.cpp 
                ${ABC_SRC}/AbcGeom/OSubD.cpp 
                ${ABC_SRC}/AbcGeom/ISubD.cpp 
                ${ABC_SRC}/AbcGeom/Visibility.cpp 
                ${ABC_SRC}/AbcGeom/XformOp.cpp 
                ${ABC_SRC}/AbcGeom/XformSample.cpp 
                ${ABC_SRC}/AbcGeom/IXform.cpp 
                ${ABC_SRC}/AbcGeom/OXform.cpp 
                ${ABC_SRC}/AbcMaterial/OMaterial.cpp 
                ${ABC_SRC}/AbcMaterial/IMaterial.cpp 
                ${ABC_SRC}/AbcMaterial/MaterialFlatten.cpp 
                ${ABC_SRC}/AbcMaterial/MaterialAssignment.cpp 
                ${ABC_SRC}/AbcMaterial/InternalUtil.cpp
)

##set (HOME "/home/OF3D/zhangjian")
##set (BOOST_ROOT ${HOME}/Documents/boost_1_51_0)
##set (BOOST_LIBRARYDIR ${BOOST_ROOT}/stage/lib)

##SET( Boost_USE_STATIC_LIBS TRUE )
##SET( Boost_USE_MULTITHREADED TRUE )
##SET( Boost_NO_BOOST_CMAKE TRUE ) 

##FIND_PACKAGE( Boost COMPONENTS system date_time regex iostreams REQUIRED thread )
##FIND_PACKAGE(ZLIB REQUIRED)

message (ZLIB_INC ${ZLIB_INCLUDE_DIR})
message (ZLIB_LIB ${ZLIB_LIBRARY})

message (Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
message (Boost_LIBRARIES ${Boost_LIBRARIES})

include_directories (${Boost_INCLUDE_DIRS})
# target_link_libraries (alembic ${Boost_LIBRARIES})

include_directories ("/usr/local/include/OpenEXR")

# set (LINK_FLAGS ${LINK_FLAGS} -fPIC -shared -Wl -lboost_regex -lboost_system -lboost_date_time -L${MAYA_LIB_PATH} -#lOpenMaya -lFoundation -lOpenMayaUI -lOpenMayaFX -L${ABC_LIB} -lAbcWFObjConvert -lAlembicAbc -lAlembicAbcCoreAbstract -#lAlembicAbcCoreFactory -lAlembicAbcCoreHDF5 -lAlembicAbcGeom -lAlembicMaterial -lAlembicOgawa -lAlembicUtill)

message (LINK_FLAGS ${LINK_FLAGS})
# set (LINK_FLAGS ${LINK_FLAGS} -lz -lbz2)

add_library(alembic STATIC ${SOURCES})

set (LINK_FLAGS ${LINK_FLAGS} -fPIC -Wl)

##set (ABCLIBDIR "/home/OF3D/zhangjian/Documents/aphid/abcstatic")
##target_link_libraries (alembic ${ABCLIBDIR}/libalembic.a
##${ABCLIBDIR}/libagawa.a)

##target_link_libraries (alembic ${MAYA_LOCATION}/devkit/Alembic/lib/libhdf5.a)

##target_link_libraries (alembic /usr/lib64/libz.so)
##target_link_libraries (alembic ${BOOST_LIBRARYDIR}/libboost_iostreams.a
##${BOOST_LIBRARYDIR}/libboost_system.a
##${BOOST_LIBRARYDIR}/libboost_date_time.a
##${BOOST_LIBRARYDIR}/libboost_regex.a
##${BOOST_LIBRARYDIR}/libboost_filesystem.a)

##set (OPENEXR_LIB ${MAYA_LIB_PATH})
##target_link_libraries (alembic ${OPENEXR_LIB}/libHalf.a
##${OPENEXR_LIB}/libIlmImf.a
##${OPENEXR_LIB}/libIlmThread.a
##${OPENEXR_LIB}/libIex.a)

##set (HDF5LIB "/home/OF3D/zhangjian/Documents/hdf5-1.8.9/hdf5/lib")
##target_link_libraries (alembic ${HDF5LIB}/libhdf5.a
##${HDF5LIB}/libhdf5_hl.a)



