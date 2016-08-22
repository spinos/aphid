# basic CMake project for ai shader
# 

cmake_minimum_required (VERSION 2.8)
project (ExtendShaders)

SET (ARNOLD_ROOT D:/usr/Arnold-4.2.4.1)
include_directories (${ARNOLD_ROOT}/include)

add_library(ExtendShaders MODULE ParamPnt2.cpp Loader.cpp)
target_link_libraries (ExtendShaders ${ARNOLD_ROOT}/lib/ai.lib)
