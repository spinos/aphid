cmake_minimum_required (VERSION 2.8)
project (SimpleRender)
# The version number.
set (SimpleRender_VERSION_MAJOR 1)
set (SimpleRender_VERSION_MINOR 1)
 
# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/SimpleRenderConfig.h.in"
  "${PROJECT_BINARY_DIR}/SimpleRenderConfig.h"
  )
 
include_directories ("D:/usr/arnoldSDK/arnold4014/include")

add_executable(SimpleRender main.cpp)

target_link_libraries (SimpleRender "D:/usr/arnoldSDK/arnold4014/lib/ai.lib")
