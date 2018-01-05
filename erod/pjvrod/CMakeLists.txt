set (PJV_HEADER_FILES window.h
glwidget.h
ElasticRodSystem.h
TestSolver.h)

set (PJV_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
${APHID_QT_DIR}/BaseSolverThread.cpp
glwidget.cpp
window.cpp
ElasticRodSystem.cpp
TestSolver.cpp
main.cpp)

ADD_EXECUTABLE (pjvt ${PJV_HEADER_FILES} ${PJV_SOURCE_FILES})

target_link_libraries (pjvt Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (pjvt ${OPENGL_LIBRARIES})

target_link_libraries (pjvt ${APHID_FOUND_LIBRARY})
target_link_libraries (pjvt ${APHID_MATH_LIBRARY})
target_link_libraries (pjvt ${APHID_GEOM_LIBRARY})
target_link_libraries (pjvt ${APHID_OGL_LIBRARY})
target_link_libraries (pjvt ${APHID_UI_LIBRARY})
target_link_libraries (pjvt ${APHID_TOPO_LIBRARY})
target_link_libraries (pjvt ${APHID_PBD_LIBRARY})
target_link_libraries (pjvt ${APHID_SDB_LIBRARY})
target_link_libraries (pjvt ${APHID_SMP_LIBRARY})

INCLUDE (${APHID_DIR}/cmake/boost_config.cmake)

target_link_libraries (pjvt ${Boost_SYSTEM_LIBRARY})

IF (WIN32)
target_link_libraries (pjvt ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (pjvt ${Boost_THREAD_LIBRARY})
target_link_libraries (pjvt ${Boost_CHRONO_LIBRARY})
ENDIF ()

target_link_libraries (pjvt ${BLAS_LIBRARY})
target_link_libraries (pjvt ${LAPACK_LIBRARY})
target_link_libraries (pjvt ${F2C_LIBRARY})
