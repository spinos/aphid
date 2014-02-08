# basic CMake project for ai dso
# 

cmake_minimum_required (VERSION 2.8)
project (simpleDSO)
# The version number.
set (SimpleDSO_VERSION_MAJOR 1)
set (SimpleDSO_VERSION_MINOR 0)
 
# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/SimpleDSOConfig.h.in"
  "${PROJECT_BINARY_DIR}/SimpleDSOConfig.h"
  )
 
# add the binary tree to the search path for include files
# so that we will find SimpleDSOConfig.h
include_directories("${PROJECT_BINARY_DIR}")
include_directories ("D:/usr/arnoldSDK/arnold4014/include")

add_library(simpleDSO MODULE proceduralMain.cpp)
target_link_libraries (simpleDSO "D:/usr/arnoldSDK/arnold4014/lib/ai.lib")
