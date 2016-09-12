cmake_minimum_required (VERSION 2.8)
project (wvlt)
# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)
set (CMAKE_BUILD_TYPE RELEASE)
set (HEADER_FILES "../shared/Morton3D.h")
include_directories ( ../shared)
add_executable (wvlt ${HEADER_FILES} main.cpp)

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
    set (APHID_DIR "D:/aphid")
ELSE ()
    set (APHID_DIR /home/td05/aphid)
ENDIF ()
message (" aphid is  " ${APHID_DIR} )

IF (UNIX)
    SET (APHID_MATH_LIBRARY ${APHID_DIR}/lib/libaphidMath.so)
    SET (APHID_GEOM_LIBRARY ${APHID_DIR}/lib/libaphidGeom.so)
    ##SET (APHID_H5_LIBRARY ${APHID_DIR}/lib/libaphidH5.so)
ELSE ()

ENDIF ()

if (WIN32)
set (Aphid_Library D:/aphid/lib/aphidMath.lib
D:/aphid/lib/aphidGeom.lib)
else ()
SET (Aphid_Library /users/jianzhang/aphid/lib/libaphidMath.a
/users/jianzhang/aphid/lib/libaphidGeom.a)
endif ()

target_link_libraries (wvlt ${APHID_MATH_LIBRARY})
target_link_libraries (wvlt ${APHID_GEOM_LIBRARY})

SET (EXECUTABLE_OUTPUT_PATH ../bin)


