set (LBM_HEADER_FILES window.h
glwidget.h)

set (LBM_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
glwidget.cpp
window.cpp
main.cpp)

ADD_EXECUTABLE (lbmt ${LBM_HEADER_FILES} ${LBM_SOURCE_FILES})

target_link_libraries (lbmt Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (lbmt ${OPENGL_LIBRARIES})

target_link_libraries (lbmt ${APHID_FOUND_LIBRARY})
target_link_libraries (lbmt ${APHID_MATH_LIBRARY})
target_link_libraries (lbmt ${APHID_GEOM_LIBRARY})
target_link_libraries (lbmt ${APHID_OGL_LIBRARY})
target_link_libraries (lbmt ${APHID_SDB_LIBRARY})
target_link_libraries (lbmt ${APHID_UI_LIBRARY})
target_link_libraries (lbmt ${APHID_TOPO_LIBRARY})
target_link_libraries (lbmt ${APHID_LBM_LIBRARY})

INCLUDE (${APHID_DIR}/cmake/boost_config.cmake)

target_link_libraries (lbmt ${Boost_SYSTEM_LIBRARY})

IF (WIN32)
target_link_libraries (lbmt ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (lbmt ${Boost_THREAD_LIBRARY})
target_link_libraries (lbmt ${Boost_CHRONO_LIBRARY})
ENDIF ()

target_link_libraries (lbmt ${BLAS_LIBRARY})
target_link_libraries (lbmt ${LAPACK_LIBRARY})
target_link_libraries (lbmt ${F2C_LIBRARY})
