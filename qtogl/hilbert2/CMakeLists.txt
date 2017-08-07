set (HIL_HEADER_FILES ${APHID_QT_DIR}/Base3DView.h 
Hilbert2D.h
window.h
widget.h)

set (HIL_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
Hilbert2D.cpp
widget.cpp
window.cpp
main.cpp)

add_executable (hilbert2 ${HIL_HEADER_FILES} ${HIL_SOURCE_FILES})

target_link_libraries (hilbert2 Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (hilbert2 ${OPENGL_LIBRARIES})
target_link_libraries (hilbert2 ${APHID_FOUND_LIBRARY})
target_link_libraries (hilbert2 ${APHID_MATH_LIBRARY})
target_link_libraries (hilbert2 ${APHID_GEOM_LIBRARY})
target_link_libraries (hilbert2 ${APHID_UI_LIBRARY})
target_link_libraries (hilbert2 ${APHID_SDB_LIBRARY})
target_link_libraries (hilbert2 ${APHID_TOPO_LIBRARY})
target_link_libraries (hilbert2 ${APHID_OGL_LIBRARY})

target_link_libraries (hilbert2 ${Boost_SYSTEM_LIBRARY})
target_link_libraries (hilbert2 ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (hilbert2 ${Boost_REGEX_LIBRARY})
target_link_libraries (hilbert2 ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (hilbert2 ${Boost_THREAD_LIBRARY})
target_link_libraries (hilbert2 ${Boost_CHRONO_LIBRARY})

IF(WIN32)
ELSEIF (APPLE)
ELSEIF (UNIX)
target_link_libraries (hilbert2 ${GLEW_LIBRARY})
target_link_libraries (hilbert2 ${BLAS_LIBRARY})

ENDIF()
