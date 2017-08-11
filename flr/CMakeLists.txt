## http://www.cmake.org/cmake/help/v2.8.12/cmake.html#module:FindQt4
## http://www.cmake.org/cmake/help/v3.0/manual/cmake-qt.7.html#introduction

cmake_minimum_required (VERSION 2.8)
project (flr)

set (CMAKE_BUILD_TYPE RelWithDebInfo)
SET (EXECUTABLE_OUTPUT_PATH ../bin)
SET (CMAKE_INCLUDE_CURRENT_DIR ON)

INCLUDE (../cmake/base_config.cmake)
INCLUDE (../cmake/aphid_config.cmake)
INCLUDE (../cmake/qt_config.cmake)
INCLUDE (../cmake/clapack_config.cmake)
INCLUDE (../cmake/boost_config.cmake)
include_directories (${APHID_DIR}/flr)

## https://cmake.org/Wiki/CMake:Bundles_And_Frameworks
## set(CMAKE_MACOSX_BUNDLE ON)

add_subdirectory(whiteNoise)

##MESSAGE ("CMAKE_EXE_LINKER_FLAGS" ${CMAKE_EXE_LINKER_FLAGS})
