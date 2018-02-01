set (SDF_HEADER_FILES ${APHID_QT_DIR}/Base3DView.h 
LegendreDFTest.h
window.h
widget.h)

set (SDF_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
LegendreDFTest.cpp
widget.cpp
window.cpp
main.cpp)

add_executable (sdft ${SDF_HEADER_FILES} ${SDF_SOURCE_FILES})

target_link_libraries (sdft Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (sdft ${OPENGL_LIBRARIES})
target_link_libraries (sdft ${APHID_FOUND_LIBRARY})
target_link_libraries (sdft ${APHID_MATH_LIBRARY})
target_link_libraries (sdft ${APHID_GEOM_LIBRARY})
target_link_libraries (sdft ${APHID_UI_LIBRARY})
target_link_libraries (sdft ${APHID_SDB_LIBRARY})
target_link_libraries (sdft ${APHID_TOPO_LIBRARY})
target_link_libraries (sdft ${APHID_OGL_LIBRARY})

target_link_libraries (sdft ${Boost_SYSTEM_LIBRARY})
target_link_libraries (sdft ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (sdft ${Boost_REGEX_LIBRARY})
target_link_libraries (sdft ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (sdft ${Boost_THREAD_LIBRARY})
target_link_libraries (sdft ${Boost_CHRONO_LIBRARY})

IF(WIN32)
ELSEIF (APPLE)
ELSEIF (UNIX)
target_link_libraries (sdft ${GLEW_LIBRARY})
target_link_libraries (sdft ${BLAS_LIBRARY})

ENDIF()
