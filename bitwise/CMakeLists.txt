cmake_minimum_required (VERSION 2.8)
project (testbit)
# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)
set (CMAKE_BUILD_TYPE RELEASE)
set (HEADER_FILES "../shared/Morton3D.h")
include_directories ( ../shared ../kdtree)
add_executable (testbit ${HEADER_FILES} main.cpp)

if(WIN32)
	set (Boost_INCLUDE_DIR "D:/usr/boost_1_51_0")
	set (Boost_LIBRARY_DIR "D:/usr/boost_1_51_0/stage/lib")
	set (Boost_USE_STATIC_LIBS ON)
	set (Boost_USE_MULTITHREADED ON)
	include_directories ("D:/usr/boost_1_51_0")
	find_package(Boost 1.51 COMPONENTS system filesystem date_time regex thread chrono)
else()

if(APPLE)
	## set (Boost_INCLUDE_DIR "/Users/jollyroger/Library/boost_1_43_0")
	## set (Boost_LIBRARY_DIR "/Users/jollyroger/Library/boost_1_43_0/stage/lib")
	SET (Boost_INCLUDE_DIR "/Users/jianzhang/Library/boost_1_55_0")
	SET (Boost_LIBRARY_DIR "/Users/jianzhang/Library/boost_1_55_0/stage/lib")
	include_directories (${Boost_INCLUDE_DIR})
	find_package(Boost 1.55 COMPONENTS system filesystem date_time regex thread)
	include_directories ("/usr/local/include")

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

target_link_libraries (testbit ${Boost_SYSTEM_LIBRARY})
target_link_libraries (testbit ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (testbit ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (testbit ${Boost_REGEX_LIBRARY})
target_link_libraries (testbit ${Boost_THREAD_LIBRARY})
target_link_libraries (testbit ${Boost_CHRONO_LIBRARY})

if (WIN32)
set (Aphid_Library D:/aphid/lib/aphidMath.lib
D:/aphid/lib/aphidGeom.lib)
else ()
## set (Aphid_Library "/users/jollyroger/aphid/lib/libaphid.a")
SET (Aphid_Library /users/jianzhang/aphid/lib/libaphidMath.a
/users/jianzhang/aphid/lib/libaphidGeom.a)
endif ()

target_link_libraries (testbit ${Aphid_Library})
