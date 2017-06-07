cmake_minimum_required (VERSION 2.8)
project (hes)
set (CMAKE_AUTOMOC ON)
set (CMAKE_AUTORCC ON)
set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_BUILD_TYPE RELEASE)

include (../cmake/base_config.cmake)
include_directories (${APHID_DIR}/hesperis)

add_subdirectory(viewer)
