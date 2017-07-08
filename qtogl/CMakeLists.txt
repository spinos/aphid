## http://www.cmake.org/cmake/help/v2.8.12/cmake.html#module:FindQt4
## http://www.cmake.org/cmake/help/v3.0/manual/cmake-qt.7.html#introduction

cmake_minimum_required (VERSION 2.8)
project (glstrudy)
set (CMAKE_AUTOMOC ON)
set (CMAKE_AUTORCC ON)
set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_BUILD_TYPE RELEASE)
SET (EXECUTABLE_OUTPUT_PATH ../bin)

INCLUDE (../cmake/base_config.cmake)
INCLUDE (../cmake/aphid_config.cmake)
INCLUDE (../cmake/ogl_config.cmake)
INCLUDE (../cmake/qt_config.cmake)
INCLUDE (../cmake/clapack_config.cmake)
INCLUDE (../cmake/boost_config.cmake)

SET (GPDF_HEADER cactus.h)

SET (GPDF_SOURCE ${APHID_QT_DIR}/Base3DView.cpp
${APHID_QT_DIR}/BaseImageWidget.cpp
${APHID_QT_DIR}/Plot1DWidget.cpp
gpdfwidget.cpp
gpdfwindow.cpp
gpdfxwidget.cpp
gpdfxdialog.cpp
gpdfmain.cpp)

add_executable (gpdeform ${GP_HEADER} ${GPDF_SOURCE})
target_link_libraries (gpdeform Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (gpdeform ${OPENGL_LIBRARIES})
target_link_libraries (gpdeform ${APHID_FOUND_LIBRARY})
target_link_libraries (gpdeform ${APHID_MATH_LIBRARY})
target_link_libraries (gpdeform ${APHID_GEOM_LIBRARY})
target_link_libraries (gpdeform ${APHID_UI_LIBRARY})
target_link_libraries (gpdeform ${APHID_SDB_LIBRARY})
target_link_libraries (gpdeform ${APHID_TOPO_LIBRARY})
target_link_libraries (gpdeform ${APHID_OGL_LIBRARY})

target_link_libraries (gpdeform ${BLAS_LIBRARY})
target_link_libraries (gpdeform ${LAPACK_LIBRARY})
target_link_libraries (gpdeform ${F2C_LIBRARY})

target_link_libraries (gpdeform ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (gpdeform ${Boost_SYSTEM_LIBRARY})

IF (WIN32)
ELSEIF (APPLE)
ELSEIF (UNIX)
target_link_libraries (gpdeform ${GLEW_LIBRARY})
ENDIF ()

## https://cmake.org/Wiki/CMake:Bundles_And_Frameworks
## set(CMAKE_MACOSX_BUNDLE ON)

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
