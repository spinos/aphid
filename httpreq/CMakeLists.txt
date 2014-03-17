cmake_minimum_required (VERSION 2.8)
project (sync_client)

add_executable(sync_client main.cpp)

if(Win32)
	set (Boost_INCLUDE_DIR "D:/usr/boost_1_51_0")
	set (Boost_LIBRARY_DIR "D:/usr/boost_1_51_0/stage/lib")
	set (Boost_USE_STATIC_LIBS ON)
	set (Boost_USE_MULTITHREADED ON)
	include_directories ("D:/usr/boost_1_51_0")
	find_package(Boost 1.51 COMPONENTS system date_time regex)
else()
	set (Boost_INCLUDE_DIR "/users/jianzhang/Library/boost_1_55_0")
	set (Boost_LIBRARY_DIR "/users/jianzhang/Library/boost_1_55_0/stage/lib")
	include_directories ("/users/jianzhang/Library/boost_1_55_0")
	find_package(Boost 1.55 COMPONENTS system date_time regex)
	include_directories ("/usr/local/include")
	target_link_libraries (sync_client "xmlrpc" "xmlrpc_abyss" "xmlrpc_client++" "xmlrpc_cpp" "xmlrpc++" "xmlrpc_client" "xmlrpc_util" "xmlrpc_xmlparse")
endif()

message ("boost system is" ${Boost_SYSTEM_LIBRARY})
message ("boost date_time is" ${Boost_DATE_TIME_LIBRARY})
message ("boost regex is" ${Boost_REGEX_LIBRARY})

target_link_libraries (sync_client ${Boost_SYSTEM_LIBRARY})
target_link_libraries (sync_client ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (sync_client ${Boost_REGEX_LIBRARY})

