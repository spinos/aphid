set (PBD_HEADER_FILES window.h
glwidget.h
SolverThread.h)

set (PBD_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
${APHID_QT_DIR}/BaseSolverThread.cpp
glwidget.cpp
window.cpp
SolverThread.cpp
main.cpp)

ADD_EXECUTABLE (rodt ${PBD_HEADER_FILES} ${PBD_SOURCE_FILES})

target_link_libraries (rodt Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (rodt ${OPENGL_LIBRARIES})

target_link_libraries (rodt ${APHID_FOUND_LIBRARY})
target_link_libraries (rodt ${APHID_MATH_LIBRARY})
target_link_libraries (rodt ${APHID_GEOM_LIBRARY})
target_link_libraries (rodt ${APHID_OGL_LIBRARY})
target_link_libraries (rodt ${APHID_UI_LIBRARY})
target_link_libraries (rodt ${APHID_TOPO_LIBRARY})
target_link_libraries (rodt ${APHID_PBD_LIBRARY})
target_link_libraries (rodt ${APHID_SDB_LIBRARY})
target_link_libraries (rodt ${APHID_SMP_LIBRARY})

INCLUDE (${APHID_DIR}/cmake/boost_config.cmake)

target_link_libraries (rodt ${Boost_SYSTEM_LIBRARY})

IF (WIN32)
target_link_libraries (rodt ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (rodt ${Boost_THREAD_LIBRARY})
target_link_libraries (rodt ${Boost_CHRONO_LIBRARY})
ENDIF ()

target_link_libraries (rodt ${BLAS_LIBRARY})
target_link_libraries (rodt ${LAPACK_LIBRARY})
target_link_libraries (rodt ${F2C_LIBRARY})
