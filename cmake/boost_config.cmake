MESSAGE (" find boost ")
if(WIN32)
set (Boost_INCLUDE_DIR d:/usr/boost_1_51_0)
set (Boost_LIBRARY_DIR d:/usr/boost_1_51_0/stage/lib)
	set (Boost_USE_STATIC_LIBS ON)
	set (Boost_USE_MULTITHREADED ON)
	include_directories ("D:/usr/boost_1_51_0")

elseif (APPLE)
set (Boost_INCLUDE_DIR ${APPLE_HOME}/Library/boost_1_55_0)
	set (Boost_LIBRARY_DIR ${APPLE_HOME}/Library/boost_1_55_0/stage/lib)
	include_directories (${APPLE_HOME}/Library/boost_1_55_0)
	set (Boost_USE_STATIC_LIBS ON)
	

ELSEIF (UNIX)
	set (Boost_INCLUDE_DIR /home/td21/usr/boost_1_51_0)
	set (BOOST_LIBRARY_DIR /home/td21/usr/boost_1_51_0/stage/lib)
endif()

if(WIN32)
    find_package(Boost 1.51 COMPONENTS system filesystem date_time regex thread chrono 
iostreams zlib)
ELSEIF (APPLE)
    find_package(Boost 1.55 REQUIRED  COMPONENTS system filesystem date_time regex thread)
else()
    find_package(Boost 1.51 COMPONENTS system filesystem date_time regex thread chrono iostreams)
endif()

message (" boost system is " ${Boost_SYSTEM_LIBRARY})
message (" boost date_time is " ${Boost_DATE_TIME_LIBRARY})
message (" boost regex is " ${Boost_REGEX_LIBRARY})
message (" boost filesystem is " ${Boost_FILESYSTEM_LIBRARY})
message (" boost thread is " ${Boost_THREAD_LIBRARY})
message (" boost chrono is " ${Boost_CHRONO_LIBRARY})
message (" boost iostreams is " ${Boost_IOSTREAMS_LIBRARY})
message (" boost zlib is " ${Boost_ZLIB_LIBRARY})

include_directories (${Boost_INCLUDE_DIR})

set (AttributeNoninline "__attribute__ ((noinline))")
add_definitions (-DBOOST_NOINLINE=${AttributeNoninline})

