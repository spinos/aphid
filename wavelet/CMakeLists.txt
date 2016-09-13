cmake_minimum_required (VERSION 2.8)
project (wvlt)
# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)
set (CMAKE_BUILD_TYPE RELEASE)
SET (CMAKE_AUTOMOC ON)
SET (CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories ( ../shared)
add_executable (wvlt main.cpp)

if(WIN32)
	set (Boost_INCLUDE_DIR "D:/usr/boost_1_51_0")
	set (Boost_LIBRARY_DIR "D:/usr/boost_1_51_0/stage/lib")
	set (Boost_USE_STATIC_LIBS ON)
	set (Boost_USE_MULTITHREADED ON)
	include_directories ("D:/usr/boost_1_51_0")
	find_package(Boost 1.51 COMPONENTS system filesystem date_time regex thread chrono)
elseif(APPLE)
	SET (Boost_INCLUDE_DIR "/Users/jianzhang/Library/boost_1_55_0")
	SET (Boost_LIBRARY_DIR "/Users/jianzhang/Library/boost_1_55_0/stage/lib")
	include_directories (${Boost_INCLUDE_DIR})
	find_package(Boost 1.55 COMPONENTS system filesystem date_time regex thread)
	include_directories ("/usr/local/include")

else()
	SET (BOOST_ROOT "~/Public/boost_1_51_0")
	find_package(Boost 1.51 COMPONENTS system filesystem date_time regex thread chrono iostreams zlib)
	include_directories (${BOOST_ROOT})
endif()

message ("boost system is" ${Boost_SYSTEM_LIBRARY})
message ("boost date_time is" ${Boost_DATE_TIME_LIBRARY})
message ("boost regex is" ${Boost_REGEX_LIBRARY})
message ("boost filesystem is" ${Boost_FILESYSTEM_LIBRARY})
message ("boost thread is" ${Boost_THREAD_LIBRARY})

target_link_libraries (wvlt ${Boost_SYSTEM_LIBRARY})
target_link_libraries (wvlt ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (wvlt ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (wvlt ${Boost_REGEX_LIBRARY})
target_link_libraries (wvlt ${Boost_THREAD_LIBRARY})
target_link_libraries (wvlt ${Boost_CHRONO_LIBRARY})

IF(WIN32)
    set (APHID_DIR D:/aphid)
ELSEIF (APPLE)
	SET (APHID_DIR /Users/jianzhang/aphid)
ELSE ()
    set (APHID_DIR /home/td05/aphid)
ENDIF ()

message (" aphid is  " ${APHID_DIR} )

IF (WIN32)
    SET (APHID_MATH_LIBRARY ${APHID_DIR}/lib/libaphidMath.so)
    SET (APHID_GEOM_LIBRARY ${APHID_DIR}/lib/libaphidGeom.so)
ELSEIF (APPLE)
	SET (APHID_MATH_LIBRARY ${APHID_DIR}/lib/libaphidMath.a)
    SET (APHID_GEOM_LIBRARY ${APHID_DIR}/lib/libaphidGeom.a)
ELSE ()
	SET (APHID_MATH_LIBRARY ${APHID_DIR}/lib/libaphidMath.so)
    SET (APHID_GEOM_LIBRARY ${APHID_DIR}/lib/libaphidGeom.so)
ENDIF ()

MESSAGE (" aphid math library " ${APHID_MATH_LIBRARY})
MESSAGE (" aphid geom library " ${APHID_GEOM_LIBRARY})

target_link_libraries (wvlt ${APHID_MATH_LIBRARY})
target_link_libraries (wvlt ${APHID_GEOM_LIBRARY})

if(WIN32)
set (QT_QMAKE_EXECUTABLE "D:/usr/qt4.8.6/bin/qmake.exe")
endif()

find_package (Qt4 REQUIRED)

message ("qt gui library is " ${QT_QTGUI_LIBRARY})

SET (SHFT_SRC ../shared/BaseImageWidget.cpp
ftmain.cpp
ftwindow.cpp)

add_executable (shoflt ${SHFT_SRC})
target_link_libraries (shoflt ${Boost_SYSTEM_LIBRARY})
target_link_libraries (shoflt ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (shoflt ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (shoflt ${Boost_REGEX_LIBRARY})
target_link_libraries (shoflt ${Boost_THREAD_LIBRARY})
target_link_libraries (shoflt ${Boost_CHRONO_LIBRARY})

target_link_libraries (shoflt ${APHID_MATH_LIBRARY})
target_link_libraries (shoflt ${APHID_GEOM_LIBRARY})

target_link_libraries (shoflt Qt4::QtGui)

SET (EXECUTABLE_OUTPUT_PATH ../bin)

