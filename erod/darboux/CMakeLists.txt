set (DAR_HEADER_FILES window.h
TestContext.h
TestConstraint.h
glwidget.h)

set (DAR_SOURCE_FILES ${APHID_QT_DIR}/Base3DView.cpp
TestContext.cpp
TestConstraint.cpp
glwidget.cpp
window.cpp
main.cpp)

ADD_EXECUTABLE (darboux ${DAR_HEADER_FILES} ${DAR_SOURCE_FILES})

target_link_libraries (darboux Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (darboux ${OPENGL_LIBRARIES})

target_link_libraries (darboux ${APHID_FOUND_LIBRARY})
target_link_libraries (darboux ${APHID_MATH_LIBRARY})
target_link_libraries (darboux ${APHID_GEOM_LIBRARY})
target_link_libraries (darboux ${APHID_OGL_LIBRARY})
target_link_libraries (darboux ${APHID_CU_LIBRARY})
target_link_libraries (darboux ${APHID_UI_LIBRARY})
target_link_libraries (darboux ${APHID_TOPO_LIBRARY})
target_link_libraries (darboux ${APHID_PBD_LIBRARY})

INCLUDE (${APHID_DIR}/cmake/boost_config.cmake)

target_link_libraries (darboux ${Boost_SYSTEM_LIBRARY})

IF (WIN32)
target_link_libraries (darboux ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (darboux ${Boost_THREAD_LIBRARY})
target_link_libraries (darboux ${Boost_CHRONO_LIBRARY})
ENDIF ()

target_link_libraries (darboux ${BLAS_LIBRARY})
target_link_libraries (darboux ${LAPACK_LIBRARY})
target_link_libraries (darboux ${F2C_LIBRARY})
