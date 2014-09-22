
cmake_minimum_required (VERSION 2.8)
project (agawa)

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

set (SOURCES ${ABC_SRC}/AbcCoreOgawa/AprImpl.cpp 
  ${ABC_SRC}/AbcCoreOgawa/ApwImpl.cpp 
   ${ABC_SRC}/AbcCoreOgawa/ArImpl.cpp 
   ${ABC_SRC}/AbcCoreOgawa/AwImpl.cpp 
   ${ABC_SRC}/AbcCoreOgawa/CprData.cpp 
   ${ABC_SRC}/AbcCoreOgawa/CprImpl.cpp 
   ${ABC_SRC}/AbcCoreOgawa/CpwData.cpp 
   ${ABC_SRC}/AbcCoreOgawa/CpwImpl.cpp 
   ${ABC_SRC}/AbcCoreOgawa/MetaDataMap.cpp 
   ${ABC_SRC}/AbcCoreOgawa/OrData.cpp 
   ${ABC_SRC}/AbcCoreOgawa/OrImpl.cpp 
   ${ABC_SRC}/AbcCoreOgawa/OwData.cpp 
   ${ABC_SRC}/AbcCoreOgawa/OwImpl.cpp 
   ${ABC_SRC}/AbcCoreOgawa/ReadUtil.cpp 
   ${ABC_SRC}/AbcCoreOgawa/ReadWrite.cpp 
   ${ABC_SRC}/AbcCoreOgawa/SprImpl.cpp 
   ${ABC_SRC}/AbcCoreOgawa/SpwImpl.cpp 
   ${ABC_SRC}/AbcCoreOgawa/StreamManager.cpp 
   ${ABC_SRC}/AbcCoreOgawa/WriteUtil.cpp 
    ${ABC_SRC}/Ogawa/IArchive.cpp 
      ${ABC_SRC}/Ogawa/IData.cpp 
      ${ABC_SRC}/Ogawa/IGroup.cpp 
      ${ABC_SRC}/Ogawa/IStreams.cpp 
      ${ABC_SRC}/Ogawa/OArchive.cpp 
      ${ABC_SRC}/Ogawa/OData.cpp 
      ${ABC_SRC}/Ogawa/OGroup.cpp 
      ${ABC_SRC}/Ogawa/OStream.cpp
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
include_directories ("/usr/local/include/OpenEXR")

# target_link_libraries (alembic ${Boost_LIBRARIES})


message (LINK_FLAGS ${LINK_FLAGS})
# set (LINK_FLAGS ${LINK_FLAGS} -lz -lbz2)

add_library(agawa STATIC ${SOURCES})

set (LINK_FLAGS ${LINK_FLAGS} -fPIC -Wl)



