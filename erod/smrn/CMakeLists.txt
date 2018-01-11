set (SMR_HEADER_FILES window.h
glwidget.h)

set (SMR_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
glwidget.cpp
window.cpp
main.cpp)

ADD_EXECUTABLE (smrt ${SMR_HEADER_FILES} ${SMR_SOURCE_FILES})

target_link_libraries (smrt Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (smrt ${OPENGL_LIBRARIES})

target_link_libraries (smrt ${APHID_FOUND_LIBRARY})
target_link_libraries (smrt ${APHID_MATH_LIBRARY})
target_link_libraries (smrt ${APHID_GEOM_LIBRARY})
target_link_libraries (smrt ${APHID_OGL_LIBRARY})
target_link_libraries (smrt ${APHID_CU_LIBRARY})
target_link_libraries (smrt ${APHID_UI_LIBRARY})
target_link_libraries (smrt ${APHID_TOPO_LIBRARY})
target_link_libraries (smrt ${APHID_PBD_LIBRARY})

INCLUDE (${APHID_DIR}/cmake/boost_config.cmake)

target_link_libraries (smrt ${Boost_SYSTEM_LIBRARY})

IF (WIN32)
target_link_libraries (smrt ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (smrt ${Boost_THREAD_LIBRARY})
target_link_libraries (smrt ${Boost_CHRONO_LIBRARY})
ENDIF ()

target_link_libraries (smrt ${BLAS_LIBRARY})
target_link_libraries (smrt ${LAPACK_LIBRARY})
target_link_libraries (smrt ${F2C_LIBRARY})
