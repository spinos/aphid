# basic CMake project by http://cmake.org/cmake/help/cmake_tutorial.html
# additional info ftp://ftp.hdfgroup.org/HDF5/current/src/unpacked/release_docs/USING_HDF5_CMake.txt
# CMake 2.8 must set native compiler on old mac like:
# CMAKE_C_COMPILER /usr/bin/gcc
# CMAKE_CXX_COMPILER /usr/bin/g++
# configure and generate xcode project of hdf5 in cmake, then build
# find_package(hdf5) doesn't for some reason


cmake_minimum_required (VERSION 2.8)
project (Tutorial)
# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)
 
# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/TutorialConfig.h.in"
  "${PROJECT_BINARY_DIR}/TutorialConfig.h"
  )
 
# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")

find_package(HDF5 REQUIRED)
 
add_executable(Tutorial main.cpp)
