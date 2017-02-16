set (TET_HEADER_FILES ${APHID_DIR}/shared/Base3DView.h 
window.h
widget.h)

set (TET_SOURCE_FILES ${APHID_DIR}/shared/Base3DView.cpp
widget.cpp
window.cpp
main.cpp)

add_executable (tetf ${TET_HEADER_FILES} ${TET_SOURCE_FILES})

target_link_libraries (tetf Qt4::QtGui Qt4::QtOpenGL)
target_link_libraries (tetf ${OPENGL_LIBRARIES})
target_link_libraries (tetf ${APHID_FOUND_LIBRARY})
target_link_libraries (tetf ${APHID_MATH_LIBRARY})
target_link_libraries (tetf ${APHID_GEOM_LIBRARY})
target_link_libraries (tetf ${APHID_UI_LIBRARY})
target_link_libraries (tetf ${APHID_SDB_LIBRARY})
target_link_libraries (tetf ${APHID_TOPO_LIBRARY})
target_link_libraries (tetf ${APHID_OGL_LIBRARY})
target_link_libraries (tetf ${APHID_GRAPH_LIBRARY})
target_link_libraries (tetf ${APHID_TTG_LIBRARY})

target_link_libraries (tetf ${Boost_SYSTEM_LIBRARY})
IF (WIN32)
target_link_libraries (tetf ${Boost_DATE_TIME_LIBRARY})
target_link_libraries (tetf ${Boost_THREAD_LIBRARY})
target_link_libraries (tetf ${Boost_CHRONO_LIBRARY})
ENDIF ()

target_link_libraries (tetf ${BLAS_LIBRARY})
target_link_libraries (tetf ${LAPACK_LIBRARY})
target_link_libraries (tetf ${F2C_LIBRARY})
