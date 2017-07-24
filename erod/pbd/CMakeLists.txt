set (PBD_HEADER_FILES window.h
glwidget.h
SolverThread.h)

set (PBD_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
${APHID_QT_DIR}/BaseSolverThread.cpp
glwidget.cpp
window.cpp
SolverThread.cpp
main.cpp)

ADD_EXECUTABLE (pbdt ${PBD_HEADER_FILES} ${PBD_SOURCE_FILES})

target_link_libraries (pbdt Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (pbdt ${OPENGL_LIBRARIES})

target_link_libraries (pbdt ${APHID_FOUND_LIBRARY})
target_link_libraries (pbdt ${APHID_MATH_LIBRARY})
target_link_libraries (pbdt ${APHID_GEOM_LIBRARY})
target_link_libraries (pbdt ${APHID_OGL_LIBRARY})
target_link_libraries (pbdt ${APHID_CU_LIBRARY})
target_link_libraries (pbdt ${APHID_UI_LIBRARY})
target_link_libraries (pbdt ${APHID_TOPO_LIBRARY})

INCLUDE (${APHID_DIR}/cmake/boost_config.cmake)

target_link_libraries (pbdt ${Boost_SYSTEM_LIBRARY})

IF (WIN32)
target_link_libraries (pbdt ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (pbdt ${Boost_THREAD_LIBRARY})
target_link_libraries (pbdt ${Boost_CHRONO_LIBRARY})
ENDIF ()
