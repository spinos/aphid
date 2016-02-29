cmake_minimum_required (VERSION 2.8)
project (testre)
SET (CMAKE_AUTOMOC ON)
SET (CMAKE_INCLUDE_CURRENT_DIR ON)
SET (CMAKE_BUILD_TYPE Release)
include_directories (../shared)

set (HEADER_FILES 
RegexUi.h Reagan.h
)

set (SOURCE_FILES 
main.cpp RegexUi.cpp Reagan.cpp)

add_executable(testre ${HEADER_FILES} ${SOURCE_FILES})

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

FIND_PACKAGE (Boost REQUIRED COMPONENTS system filesystem date_time regex)	
	
MESSAGE (" boost version is " ${Boost_VERSION})
MESSAGE (" boost include is " ${Boost_INCLUDE_DIR} )
MESSAGE (" boost system is " ${Boost_SYSTEM_LIBRARY})
MESSAGE (" boost date_time is " ${Boost_DATE_TIME_LIBRARY})
MESSAGE (" boost regex is " ${Boost_REGEX_LIBRARY})
MESSAGE (" boost filesystem is " ${Boost_FILESYSTEM_LIBRARY})

include_directories (${Boost_INCLUDE_DIR})
	
if(WIN32)
## set (QT_QMAKE_EXECUTABLE "D:/usr/Qtx64/4.7.1/bin/qmake.exe")
set (QT_QMAKE_EXECUTABLE "D:/usr/qt4.8.6/bin/qmake.exe")
endif()

find_package (Qt4 REQUIRED)
message ("qt opengl " ${QT_QTOPENGL_LIBRARY})
SET (QT_USE_QTOPENGL ON)

target_link_libraries (testre Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries(testre ${Boost_LIBRARIES})
target_link_libraries(testre ${Aphid_Library})

