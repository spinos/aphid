set (NACA4_HEADER_FILES ${APHID_QT_DIR}/Base3DView.h 
paramdialog.h
paramwidget.h
window.h
widget.h)

set (NACA4_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
${APHID_QT_DIR}/QDoubleEditSlider.cpp 
paramdialog.cpp
paramwidget.cpp
widget.cpp
window.cpp
main.cpp)

add_executable (naca4digit ${NACA4_HEADER_FILES} ${NACA4_SOURCE_FILES})

target_link_libraries (naca4digit Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (naca4digit ${OPENGL_LIBRARIES})
target_link_libraries (naca4digit ${APHID_FOUND_LIBRARY})
target_link_libraries (naca4digit ${APHID_MATH_LIBRARY})
target_link_libraries (naca4digit ${APHID_GEOM_LIBRARY})
target_link_libraries (naca4digit ${APHID_UI_LIBRARY})
target_link_libraries (naca4digit ${APHID_SDB_LIBRARY})
target_link_libraries (naca4digit ${APHID_TOPO_LIBRARY})
target_link_libraries (naca4digit ${APHID_OGL_LIBRARY})

target_link_libraries (naca4digit ${Boost_SYSTEM_LIBRARY})
target_link_libraries (naca4digit ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (naca4digit ${Boost_REGEX_LIBRARY})
target_link_libraries (naca4digit ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries (naca4digit ${Boost_THREAD_LIBRARY})
target_link_libraries (naca4digit ${Boost_CHRONO_LIBRARY})

IF(WIN32)
ELSEIF (APPLE)
ELSEIF (UNIX)
target_link_libraries (naca4digit ${GLEW_LIBRARY})
target_link_libraries (naca4digit ${BLAS_LIBRARY})

ENDIF()
