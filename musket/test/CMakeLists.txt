# basic CMake project for ai shader
# 

cmake_minimum_required (VERSION 2.8)
project (test)

IF (WIN32)
ELSE()
  include_directories ("/usr/local/include")
ENDIF()

include_directories ("../")
link_directories ("../Release")
add_executable (test main.cpp)
target_link_libraries (test "shotgunAPI" "xmlrpc" "xmlrpc_abyss" "xmlrpc_client++" "xmlrpc_cpp" "xmlrpc++" "xmlrpc_client" "xmlrpc_util" "xmlrpc_xmlparse")
