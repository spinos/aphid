INCLUDE (${APHID_DIR}/cmake/hdf5_config.cmake)
INCLUDE (${APHID_DIR}/cmake/exr_config.cmake)

set (TTRI_HEADER_FILES ${APHID_QT_DIR}/Base3DView.h 
window.h
widget.h
Parameter.h)

set (TTRI_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
Parameter.cpp
widget.cpp
window.cpp
main.cpp)

set(CMAKE_MACOSX_BUNDLE OFF)

add_executable (ttri ${TTRI_HEADER_FILES} ${TTRI_SOURCE_FILES})

target_link_libraries (ttri Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (ttri ${OPENGL_LIBRARIES})
target_link_libraries (ttri ${APHID_FOUND_LIBRARY})
target_link_libraries (ttri ${APHID_MATH_LIBRARY})
target_link_libraries (ttri ${APHID_GEOM_LIBRARY})
target_link_libraries (ttri ${APHID_UI_LIBRARY})
target_link_libraries (ttri ${APHID_SDB_LIBRARY})
target_link_libraries (ttri ${APHID_TOPO_LIBRARY})
target_link_libraries (ttri ${APHID_OGL_LIBRARY})
target_link_libraries (ttri ${APHID_GRAPH_LIBRARY})
target_link_libraries (ttri ${APHID_TTG_LIBRARY})
target_link_libraries (ttri ${APHID_KD_LIBRARY})
target_link_libraries (ttri ${APHID_H5_LIBRARY})
target_link_libraries (ttri ${APHID_IMG_LIBRARY})

target_link_libraries (ttri ${Boost_SYSTEM_LIBRARY})
target_link_libraries (ttri ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (ttri ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (ttri ${Boost_REGEX_LIBRARY})

IF (WIN32)
target_link_libraries (ttri ${Boost_THREAD_LIBRARY})
target_link_libraries (ttri ${Boost_CHRONO_LIBRARY})

ENDIF ()

target_link_libraries (ttri ${HDF5_LIBRARIES})
target_link_libraries (ttri ${ZLIB_LIBRARY})

IF (WIN32)
target_link_libraries (ttri ${SZIP_LIBRARY})

ELSEIF (APPLE)
ELSEIF (UNIX)
target_link_libraries (ttri ${GLEW_LIBRARY})

ENDIF()

target_link_libraries (ttri ${BLAS_LIBRARY})
target_link_libraries (ttri ${LAPACK_LIBRARY})
target_link_libraries (ttri ${OpenEXR_Library})

