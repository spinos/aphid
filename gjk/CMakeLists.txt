cmake_minimum_required (VERSION 2.8)
project (gjk)
# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)

add_executable(gjk main.cpp)

if(WIN32)
	set (Boost_INCLUDE_DIR "D:/usr/boost_1_51_0")
	set (Boost_LIBRARY_DIR "D:/usr/boost_1_51_0/stage/lib")
	set (Boost_USE_STATIC_LIBS ON)
	set (Boost_USE_MULTITHREADED ON)
	include_directories ("D:/usr/boost_1_51_0")
	find_package(Boost 1.51 COMPONENTS system filesystem date_time regex)
	
	set (Aphid_INCLUDE_DIR "D:/aphid/shared")
	set (Aphid_LIBRARY_DIR "D:/aphid/lib")
else()

if(APPLE)
	set (Boost_INCLUDE_DIR "/users/jianzhang/Library/boost_1_55_0")
	set (Boost_LIBRARY_DIR "/users/jianzhang/Library/boost_1_55_0/stage/lib")
	include_directories ("/users/jianzhang/Library/boost_1_55_0")
	find_package(Boost 1.55 COMPONENTS system filesystem date_time regex)
	include_directories ("/usr/local/include")

	set (Aphid_INCLUDE_DIR "/users/jianzhang/aphid/shared")
	set (Aphid_LIBRARY_DIR "/users/jianzhang/aphid/lib")
else()
	set (Boost_INCLUDE_DIR "/home/OF3D/zhangjian/Documents/boost_1_51_0")
	set (Boost_USE_STATIC_LIBS ON)
	set (Boost_USE_MULTITHREADED ON)
	include_directories ("/home/OF3D/zhangjian/Documents/boost_1_51_0")
	find_package(Boost 1.51 COMPONENTS system filesystem date_time regex)
	
	set (Aphid_INCLUDE_DIR "/home/OF3D/zhangjian/Documents/aphid/shared")
	set (Aphid_LIBRARY_DIR "/home/OF3D/zhangjian/Documents/aphid/lib")
	
endif()
	
endif()

include_directories (${Aphid_INCLUDE_DIR})

find_library(Aphid aphid ${Aphid_LIBRARY_DIR})
message ("aphid library is" ${Aphid})

message ("boost system is" ${Boost_SYSTEM_LIBRARY})
message ("boost date_time is" ${Boost_DATE_TIME_LIBRARY})
message ("boost regex is" ${Boost_REGEX_LIBRARY})
message ("boost filesystem is" ${Boost_FILESYSTEM_LIBRARY})

target_link_libraries (gjk ${Boost_SYSTEM_LIBRARY})
target_link_libraries (gjk ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (gjk ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (gjk ${Boost_REGEX_LIBRARY})
target_link_libraries (gjk ${Aphid})


