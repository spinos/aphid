## http://www.cmake.org/cmake/help/v2.8.12/cmake.html#module:FindQt4
## http://www.cmake.org/cmake/help/v3.0/manual/cmake-qt.7.html#introduction

cmake_minimum_required (VERSION 2.8)
project (glstudy)

set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_BUILD_TYPE RelWithDebInfo)
SET (EXECUTABLE_OUTPUT_PATH ../bin)

INCLUDE (../cmake/base_config.cmake)
INCLUDE (../cmake/aphid_config.cmake)
INCLUDE (../cmake/ogl_config.cmake)
INCLUDE (../cmake/qt_config.cmake)
INCLUDE (../cmake/clapack_config.cmake)
INCLUDE (../cmake/boost_config.cmake)

## https://cmake.org/Wiki/CMake:Bundles_And_Frameworks
## set(CMAKE_MACOSX_BUNDLE ON)

add_subdirectory(hilbert2)
add_subdirectory(gpdeform)
add_subdirectory(rott)
add_subdirectory(naca4)
add_subdirectory(hermite)
add_subdirectory(pca)
add_subdirectory(hull)
add_subdirectory(cluster)
add_subdirectory(inst)
##add_subdirectory(tetf)
add_subdirectory(abg)
add_subdirectory(tkm)
add_subdirectory(ttri)
add_subdirectory(exrs)
add_subdirectory(wbg)
add_subdirectory(tvox)
add_subdirectory(gar)

##MESSAGE ("CMAKE_EXE_LINKER_FLAGS" ${CMAKE_EXE_LINKER_FLAGS})
