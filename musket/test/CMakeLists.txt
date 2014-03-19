# basic CMake project for ai shader
# 

cmake_minimum_required (VERSION 2.8)
project (test)

IF (WIN32)
	include_directories ("D:/shotguncpp/lib" "D:/usr/xmlrpc-c-1.25.26/include")
	link_directories ("D:/shotguncpp/lib" "D:/usr/xmlrpc-c-1.25.26/lib")
ELSE()
  	include_directories ("/usr/local/include" "../")
	link_directories ("../Release")
ENDIF()

add_executable (test main.cpp)

IF (WIN32)
	target_link_libraries (test "shotgunAPI" "xmlrpc" "abyss" "xmlrpccpp" "xmlrpc_misc" "xmlparse" "xmltok" "xmlhttpsys" "wininet" "wsock32")
ELSE()
	target_link_libraries (test "shotgunAPI" "xmlrpc" "xmlrpc_abyss" "xmlrpc_client++" "xmlrpc_cpp" "xmlrpc++" "xmlrpc_client" "xmlrpc_util" "xmlrpc_xmlparse")
ENDIF()
