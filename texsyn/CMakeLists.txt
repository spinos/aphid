cmake_minimum_required (VERSION 2.8)
project (txsn)
# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)
set (CMAKE_AUTOMOC ON)
set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_BUILD_TYPE RELEASE)

include_directories ( ../shared ../kdtree ../lfr)

set (HEADER_FILES ../shared/BaseImageWidget.h
../lfr/clapackTempl.h
../lfr/linearMath.h
../lfr/regr.h
../lfr/psnr.h
../lfr/dctmn.h
../lfr/LfParameter.h
../lfr/LfMachine.h
../lfr/LfThread.h
../lfr/StatisticDialog.h
../lfr/DictionaryDialog.h
../lfr/DictionaryView.h
../lfr/ImageDialog.h
TsParameter.h
TsWindow.h)

add_executable (txsn ${HEADER_FILES}
../shared/BaseImageWidget.cpp
../lfr/LfParameter.cpp
../lfr/LfThread.cpp
../lfr/LfMachine.cpp
../lfr/StatisticDialog.cpp
../lfr/DictionaryDialog.cpp
../lfr/DictionaryView.cpp
../lfr/ImageDialog.cpp
TsParameter.cpp
TsWindow.cpp
main.cpp)

if(WIN32)
    include_directories (D:/usr/CLAPACK/include)
    set(BLAS_LIBRARY D:/usr/CLAPACK/lib/blas.lib)
	set(LAPACK_LIBRARY D:/usr/CLAPACK/lib/lapack.lib)
    set(F2C_LIBRARY D:/usr/CLAPACK/lib/libf2c.lib)
    
endif()

if(APPLE)
	set(F2C_LIBRARY libf2c.a)
	set(BLAS_LIBRARY libblas.a)
	set(LAPACK_LIBRARY liblapack.a)

else ()
include_directories (/usr/local/include)
	set(BLAS_LIBRARY /usr/local/lib/libblas.a)
	set(LAPACK_LIBRARY /usr/local/lib/liblapack.a)
	set(F2C_LIBRARY /usr/local/libf2c.a)

endif()

message ("blas " ${BLAS_LIBRARY})
message ("lapack " ${LAPACK_LIBRARY})
message ("f2c " ${F2C_LIBRARY})

target_link_libraries (txsn ${F2C_LIBRARY})
target_link_libraries (txsn ${BLAS_LIBRARY})
target_link_libraries (txsn ${LAPACK_LIBRARY})

if (WIN32)
add_definitions (-DOPENEXR_DLL)
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
target_link_libraries (txsn ${OpenEXR_Library})
message (" openexr " ${OpenEXR_Library})

if (WIN32)
SET (ZLIB_DIR "D:/usr/zlib-1.2.5")
    set (ZLIB_INCLUDE_DIR d:/usr/zlib-1.2.5)
    set (ZLIB_LIBRARY d:/usr/zlib-1.2.5/zdll.lib)
endif ()

find_package (ZLIB REQUIRED)
MESSAGE (" zlib version major is " ${ZLIB_VERSION_MAJOR})
MESSAGE (" zlib version minor is " ${ZLIB_VERSION_MINOR})
MESSAGE (" zlib include is " ${ZLIB_INCLUDE_DIR})
MESSAGE (" zlib library is " ${ZLIB_LIBRARY})

target_link_libraries (txsn ${ZLIB_LIBRARY})

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
target_link_libraries (txsn ${SZIP_LIBRARIES})
endif ()

include_directories (${HDF5_INCLUDE_DIR})
target_link_libraries (txsn ${HDF5_LIBRARIES})

if(WIN32)
set (Boost_INCLUDE_DIR d:/usr/boost_1_51_0)
set (Boost_LIBRARY_DIR d:/usr/boost_1_51_0/stage/lib)
	set (Boost_USE_STATIC_LIBS ON)
	set (Boost_USE_MULTITHREADED ON)
	include_directories ("D:/usr/boost_1_51_0")

elseif (APPLE)
set (HOME "/home/OF3D/zhangjian")
set (BOOST_ROOT ${HOME}/Documents/boost_1_51_0)
set (BOOST_LIBRARYDIR ${BOOST_ROOT}/stage/lib)
SET( Boost_USE_STATIC_LIBS TRUE )
SET( Boost_USE_MULTITHREADED TRUE )
SET( Boost_NO_BOOST_CMAKE TRUE ) 

ELSEIF (UNIX)
	SET (BOOST_ROOT "~/Public/boost_1_51_0")
endif()

if(WIN32)
    find_package(Boost 1.51 COMPONENTS system filesystem date_time regex thread chrono 
iostreams zlib)
else()
    find_package(Boost 1.51 COMPONENTS system filesystem date_time regex thread chrono iostreams zlib)
endif()

message (" boost system is " ${Boost_SYSTEM_LIBRARY})
message (" boost date_time is " ${Boost_DATE_TIME_LIBRARY})
message (" boost regex is " ${Boost_REGEX_LIBRARY})
message (" boost filesystem is " ${Boost_FILESYSTEM_LIBRARY})
message (" boost thread is " ${Boost_THREAD_LIBRARY})
message (" boost chrono is " ${Boost_CHRONO_LIBRARY})
message (" boost iostreams is " ${Boost_IOSTREAMS_LIBRARY})
message (" boost zlib is " ${Boost_ZLIB_LIBRARY})

ADD_DEFINITIONS (-DNOMINMAX )
ADD_DEFINITIONS (-DBOOST_PREVENT_MACRO_SUBSTITUTION)
ADD_DEFINITIONS (-DBOOST_USE_STD_MIN )
ADD_DEFINITIONS (-DBOOST_USE_STD_MAX )
#ADD_DEFINITIONS (-DBOOST_NO_STD_MIN_MAX)

include_directories (${BOOST_ROOT})

if(WIN32)
else()
set (LINK_FLAGS ${LINK_FLAGS} -shared -fPIC -Wl)
endif()

set (AttributeNoninline "__attribute__ ((noinline))")
add_definitions (-DBOOST_NOINLINE=${AttributeNoninline})



target_link_libraries (txsn ${Boost_SYSTEM_LIBRARY})
target_link_libraries (txsn ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (txsn ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (txsn ${Boost_REGEX_LIBRARY})
target_link_libraries (txsn ${Boost_THREAD_LIBRARY})
target_link_libraries (txsn ${Boost_CHRONO_LIBRARY})

if(WIN32)
set (QT_QMAKE_EXECUTABLE "D:/usr/qt4.8.6/bin/qmake.exe")
endif()

find_package (Qt4 REQUIRED)
target_link_libraries (txsn Qt4::QtGui)
## message ("qt opengl " ${QT_QTOPENGL_LIBRARY})
## set (QT_USE_QTOPENGL ON)
## target_link_libraries (txsn Qt4::QtGui Qt4::QtOpenGL)

if (WIN32)
set (Aphid_Math_Library "D:/aphid/lib/aphidMath.lib")
set (Aphid_Img_Library "D:/aphid/lib/aphidImg.lib")
else ()
set (Aphid_Math_Library /users/jianzhang/aphid/lib/libaphidMath.a)
set (Aphid_Img_Library /users/jianzhang/aphid/lib/libaphidImg.a)
endif ()

target_link_libraries (txsn ${Aphid_Math_Library})
target_link_libraries (txsn ${Aphid_Img_Library})

SET (EXECUTABLE_OUTPUT_PATH ../bin)

