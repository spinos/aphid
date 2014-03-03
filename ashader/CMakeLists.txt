# basic CMake project for ai shader
# 

cmake_minimum_required (VERSION 2.8)
project (ExtendShaders)

include_directories ("D:/usr/arnoldSDK/arnold4014/include")

SET (CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS_INIT} "/machine:X64")
    
add_library(ExtendShaders MODULE ParamPnt2.cpp Loader.cpp)
target_link_libraries (ExtendShaders "D:/usr/arnoldSDK/arnold4014/lib/ai.lib")
