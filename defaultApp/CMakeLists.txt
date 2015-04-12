## http://www.cmake.org/cmake/help/v2.8.12/cmake.html#module:FindQt4
## http://www.cmake.org/cmake/help/v3.0/manual/cmake-qt.7.html#introduction

cmake_minimum_required (VERSION 2.8)
project (foo)
set (CMAKE_AUTOMOC ON)
set (CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories ("../shared")

set (HEADER_FILES "../shared/Base3DView.h" 
"window.h"
"glwidget.h"
"../shared/Morton3D.h")

set (SOURCE_FILES "../shared/Base3DView.cpp"
"glwidget.cpp"
"window.cpp"
"main.cpp")

add_executable (foo ${HEADER_FILES} ${SOURCE_FILES})

find_package (OpenGL REQUIRED)
target_link_libraries (foo ${OPENGL_LIBRARIES})

set (Aphid_Library "/users/jianzhang/aphid/lib/libaphid.a")
target_link_libraries (foo ${Aphid_Library})

set (OpenEXR_Library "/usr/local/lib/libHalf.a"
"/usr/local/lib/libIex.a"
"/usr/local/lib/libIlmImf.a"
"/usr/local/lib/libIlmThread.a")
target_link_libraries (foo ${OpenEXR_Library})

find_package (zlib REQUIRED)
target_link_libraries (foo ${ZLIB_LIBRARIES})

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
	find_package(Boost 1.51 COMPONENTS system filesystem date_time regex thread)
else()

if(APPLE)
	set (Boost_INCLUDE_DIR "/users/jianzhang/Library/boost_1_55_0")
	set (Boost_LIBRARY_DIR "/users/jianzhang/Library/boost_1_55_0/stage/lib")
	include_directories ("/users/jianzhang/Library/boost_1_55_0")
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

target_link_libraries (foo ${Boost_SYSTEM_LIBRARY})
target_link_libraries (foo ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (foo ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (foo ${Boost_REGEX_LIBRARY})
target_link_libraries (foo ${Boost_THREAD_LIBRARY})

